"""
MLLM Hallucination Evaluation Pipeline

Multi-GPU data parallelism: each GPU loads a full copy of the model and
processes a different subset of samples in parallel.

Usage:
    # Single GPU
    python main.py --benchmark videomme --model qwen25_vl --eval_method baseline \
        --video_dir /path/to/videos --num_gpus 1

    # 4 GPUs (4x speed), EgoSchema
    python main.py --benchmark egoschema --model qwen25_vl \
        --eval_method remove_correct_add_none \
        --video_dir /path/to/videos --num_gpus 4

    # With config file
    python main.py --config configs/default.yaml
"""

import argparse
import json
import logging
import os
import sys

import torch
import torch.multiprocessing as mp
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks import BENCHMARKS, build_benchmark
from eval import EVALUATORS
from utils.logger import setup_logging


DEFAULT_MODEL_PATHS = {
    "llava_video": "lmms-lab/LLaVA-Video-7B-Qwen2",
    "qwen2_vl": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen25_vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3_vl": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen25_omni": "Qwen/Qwen2.5-Omni-7B",
    "qwen3_omni": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "gemma3_4b": "google/gemma-3-4b-it",
    "gemma3_12b": "google/gemma-3-12b-it",
    "gemma3_27b": "google/gemma-3-27b-it",
    "api_models": "gemini-2.5-flash",
    "internvl2_5": "OpenGVLab/InternVL2_5-8B",
    "internvl3": "OpenGVLab/InternVL3-8B",
    "internvl3_5": "OpenGVLab/InternVL3_5-8B",
    "video_r1": "Video-R1/Video-R1-7B",
    "videochat_flash": "OpenGVLab/VideoChat-Flash-Qwen2-7B_res448",
    "mimo_vl": "XiaomiMiMo/MiMo-VL-7B-RL",
    "keye_vl": "Kwai-Keye/Keye-VL-1_5-8B",
    "vllm": "",  # User must specify --model_path
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="MLLM Hallucination Evaluation Pipeline"
    )
    parser.add_argument("--config", type=str, default=None)

    # Benchmark
    parser.add_argument("--benchmark", type=str, default=None,
                        choices=list(BENCHMARKS.keys()))
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--anno_path", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)

    # Model
    parser.add_argument("--model", type=str, default=None,
                        choices=["llava_video", "qwen2_vl", "qwen25_vl",
                                 "qwen3_vl", "qwen25_omni", "qwen3_omni",
                                 "gemma3_12b", "gemma3_4b", "gemma3_27b",
                                 "api_models", "internvl2_5", "internvl3", "internvl3_5",
                                 "video_r1", "videochat_flash", "mimo_vl", "keye_vl",
                                 "vllm"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--api_base", type=str, default=None,
                        help="API base URL for api_models (e.g. http://host:4000)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key for api_models")
    parser.add_argument("--max_image_long_side", type=int, default=None,
                        help="Resize frames so the longest edge <= this many "
                             "pixels for api_models (e.g. 512 for gpt-4o). "
                             "0/None disables resizing.")
    parser.add_argument("--reasoning_effort", type=str, default=None,
                        help="Override reasoning_effort for api_models "
                             "(none/low/medium/high). Default: auto per model.")

    # vLLM
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                        help="vLLM tensor parallel size (default: num_gpus)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=None,
                        help="vLLM GPU memory utilization (default: 0.9)")
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="vLLM max model sequence length")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for vLLM batch inference (default: 32)")

    # Evaluation
    parser.add_argument("--eval_method", type=str, default=None,
                        choices=list(EVALUATORS.keys()))
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--use_logprobs", action="store_true")
    parser.add_argument("--no_resume", action="store_true")

    # Multi-GPU
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs for data parallelism (default: all visible)")

    # Output
    parser.add_argument("--output_dir", type=str, default=None)

    return parser.parse_args()


def merge_config(args):
    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    bench_cfg = cfg.get("benchmark", {})
    model_cfg = cfg.get("model", {})
    eval_cfg = cfg.get("eval", {})

    benchmark_name = args.benchmark or bench_cfg.get("name", "videomme")
    model_name = args.model or model_cfg.get("name", "qwen25_vl")

    # Auto-detect GPUs if not specified
    num_gpus = args.num_gpus or eval_cfg.get("num_gpus") or torch.cuda.device_count()

    config = {
        "benchmark": {
            "name": benchmark_name,
            "video_dir": args.video_dir or bench_cfg.get("video_dir", ""),
            "anno_path": args.anno_path or bench_cfg.get("anno_path"),
            "split": args.split or bench_cfg.get("split", "test"),
        },
        "model": {
            "name": model_name,
            "model_path": (args.model_path
                           or model_cfg.get("model_path")
                           or DEFAULT_MODEL_PATHS.get(model_name, "")),
            "max_frames": args.max_frames or model_cfg.get("max_frames", 64),
            "api_base": args.api_base or model_cfg.get("api_base"),
            "api_key": args.api_key or model_cfg.get("api_key"),
            "max_image_long_side": (args.max_image_long_side
                                    if args.max_image_long_side is not None
                                    else model_cfg.get("max_image_long_side")),
            "reasoning_effort": (args.reasoning_effort
                                 or model_cfg.get("reasoning_effort")),
        },
        "eval": {
            "method": args.eval_method or eval_cfg.get("method", "baseline"),
            "max_new_tokens": args.max_new_tokens or eval_cfg.get("max_new_tokens", 128),
            "use_logprobs": args.use_logprobs or eval_cfg.get("use_logprobs", False),
            "resume": not args.no_resume,
            "num_gpus": num_gpus,
        },
        "output_dir": (args.output_dir
                       or eval_cfg.get("output_dir")
                       or f"outputs/{benchmark_name}/{model_name}/{args.eval_method or eval_cfg.get('method', 'baseline')}"),
    }
    return config


def main():
    # Required for multi-process model loading
    mp.set_start_method("spawn", force=True)

    args = parse_args()
    cfg = merge_config(args)

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(log_file=os.path.join(output_dir, "eval.log"))
    logger = logging.getLogger(__name__)

    logger.info(f"Config: {json.dumps(cfg, indent=2)}")
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Build benchmark (CPU only, shared across all workers)
    bench_cfg = cfg["benchmark"]
    logger.info(f"Loading benchmark: {bench_cfg['name']}")
    bench_kwargs = {"video_dir": bench_cfg["video_dir"]}
    if bench_cfg.get("anno_path"):
        bench_kwargs["anno_path"] = bench_cfg["anno_path"]
    if bench_cfg.get("split"):
        bench_kwargs["split"] = bench_cfg["split"]
    benchmark = build_benchmark(bench_cfg["name"], **bench_kwargs)
    benchmark.load()
    logger.info(f"Loaded {len(benchmark)} samples")

    # Build evaluator (model is loaded inside each worker process)
    model_cfg = cfg["model"]
    eval_cfg = cfg["eval"]
    eval_method = eval_cfg["method"]
    num_gpus = eval_cfg["num_gpus"]

    logger.info(f"Model: {model_cfg['name']} ({model_cfg['model_path']})")
    logger.info(f"Eval method: {eval_method}, GPUs: {num_gpus}")

    evaluator_cls = EVALUATORS[eval_method]

    # Extra kwargs for API-based models
    model_kwargs = {}
    if model_cfg.get("api_base"):
        model_kwargs["api_base"] = model_cfg["api_base"]
    if model_cfg.get("api_key"):
        model_kwargs["api_key"] = model_cfg["api_key"]
    if model_cfg.get("max_image_long_side") is not None:
        model_kwargs["max_image_long_side"] = model_cfg["max_image_long_side"]
    if model_cfg.get("reasoning_effort"):
        model_kwargs["reasoning_effort"] = model_cfg["reasoning_effort"]

    # vLLM kwargs
    if model_cfg["name"] == "vllm":
        tp = (args.tensor_parallel_size
              or model_cfg.get("tensor_parallel_size")
              or num_gpus)
        model_kwargs["tensor_parallel_size"] = tp
        gpu_mem = (args.gpu_memory_utilization
                   or model_cfg.get("gpu_memory_utilization")
                   or 0.9)
        model_kwargs["gpu_memory_utilization"] = gpu_mem
        max_model_len = (args.max_model_len
                         or model_cfg.get("max_model_len"))
        if max_model_len:
            model_kwargs["max_model_len"] = max_model_len
        batch_size = (args.batch_size
                      or model_cfg.get("batch_size")
                      or 32)
        model_kwargs["batch_size"] = batch_size

    eval_kwargs = {
        "model_name": model_cfg["name"],
        "model_path": model_cfg["model_path"],
        "max_frames": model_cfg["max_frames"],
        "benchmark": benchmark,
        "output_dir": output_dir,
        "max_new_tokens": eval_cfg["max_new_tokens"],
        "resume": eval_cfg["resume"],
        "num_gpus": num_gpus,
        "model_kwargs": model_kwargs,
    }

    evaluator = evaluator_cls(**eval_kwargs)

    # Run
    logger.info("Starting inference...")
    results = evaluator.run(use_logprobs=eval_cfg["use_logprobs"])
    logger.info(f"Done. {len(results)} samples.")

    results_path = os.path.join(output_dir, f"{evaluator.name}_results.jsonl")
    print("\n" + "=" * 60)
    print(f"Benchmark: {bench_cfg['name']}")
    print(f"Model: {model_cfg['name']}")
    print(f"Eval: {eval_method}  |  GPUs: {num_gpus}")
    print(f"Samples: {len(results)} / {len(benchmark)}")
    print(f"Results: {results_path}")
    print("Metrics: run postprocess/compute_metrics.py on the results directory.")
    print("=" * 60)

if __name__ == "__main__":
    main()
