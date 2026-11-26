import os
import json
import logging
import signal
import sys
import traceback
from abc import ABC, abstractmethod

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from benchmarks.base import MCQSample

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """Base class for the evaluation settings.

    An evaluator only builds the prompt for each sample (``prepare_sample``)
    and saves the raw model output. Answer parsing and metric computation are
    done afterwards by ``postprocess/compute_metrics.py``.

    Supports multi-GPU data parallelism: each GPU loads a full copy of the
    model and processes a different subset of samples in parallel.

    Results are appended per sample to ``{output_dir}/{name}_results.jsonl``,
    so an interrupted run can resume.
    """

    name: str = ""

    def __init__(self, model_name: str, model_path: str, max_frames: int,
                 benchmark, output_dir: str,
                 max_new_tokens: int = 128, resume: bool = True,
                 num_gpus: int = 1, model_kwargs: dict = None):
        self.model_name = model_name
        self.model_path = model_path
        self.max_frames = max_frames
        self.model_kwargs = model_kwargs or {}
        self.benchmark = benchmark
        self.output_dir = output_dir
        self.max_new_tokens = max_new_tokens
        self.resume = resume
        self.num_gpus = num_gpus
        os.makedirs(output_dir, exist_ok=True)

    @abstractmethod
    def prepare_sample(self, sample: MCQSample) -> dict:
        """Transform a sample for this evaluation method.

        Returns dict with: prompt, options, expected_answer, extra.
        """
        ...

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self, use_logprobs: bool = False) -> list[dict]:
        """Run inference and return all saved results.

        Spawns one worker per GPU for data parallelism (or uses vLLM batching).
        """
        results_path = os.path.join(self.output_dir, f"{self.name}_results.jsonl")

        # Load already-completed sample ids for resume
        completed_ids = set()
        if self.resume and os.path.exists(results_path):
            with open(results_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        completed_ids.add(r["sample_id"])
            logger.info(f"Resuming: {len(completed_ids)} samples already completed")

        # Filter to remaining samples
        remaining = [s for s in self.benchmark.samples
                     if s.sample_id not in completed_ids]
        logger.info(f"Samples to process: {len(remaining)} "
                     f"(total={len(self.benchmark.samples)}, done={len(completed_ids)})")

        if not remaining:
            logger.info("All samples already completed.")
            return self._load_all_results(results_path)

        # Check if model supports batch inference (vLLM)
        if self.model_name == "vllm":
            self._batch_run(
                samples=remaining,
                results_path=results_path,
                use_logprobs=use_logprobs,
            )
            return self._load_all_results(results_path)

        if self.num_gpus <= 1:
            # Single GPU: run in main process
            self._worker_run(
                rank=0,
                samples=remaining,
                results_path=results_path,
                use_logprobs=use_logprobs,
            )
        else:
            # Multi-GPU: split samples and spawn workers
            chunks = self._split_samples(remaining, self.num_gpus)
            processes = []
            for rank in range(self.num_gpus):
                if not chunks[rank]:
                    continue
                p = mp.Process(
                    target=self._worker_run,
                    args=(rank, chunks[rank], results_path, use_logprobs),
                )
                p.start()
                processes.append((rank, p))

            # Wait for all workers and check exit status
            failed_workers = []
            for rank, p in processes:
                p.join()
                if p.exitcode != 0:
                    if p.exitcode is not None and p.exitcode < 0:
                        sig = -p.exitcode
                        sig_name = signal.Signals(sig).name if sig in signal.Signals._value2member_map_ else f"signal {sig}"
                        logger.error(
                            f"[{self.name}|GPU{rank}] Worker killed by {sig_name} "
                            f"(exit code {p.exitcode}). "
                            f"Common causes: OOM (SIGKILL=9), segfault (SIGSEGV=11)."
                        )
                    else:
                        logger.error(
                            f"[{self.name}|GPU{rank}] Worker exited with code {p.exitcode}. "
                            f"Check eval.log for traceback."
                        )
                    failed_workers.append(rank)

            if failed_workers:
                logger.error(
                    f"{len(failed_workers)}/{len(processes)} worker(s) failed "
                    f"(GPU ranks: {failed_workers}). "
                    f"Results may be incomplete — re-run with --no_resume disabled to retry failed samples."
                )

        return self._load_all_results(results_path)

    # ------------------------------------------------------------------
    # Result record (raw model output, no answer parsing)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_result(sample: MCQSample, prepared: dict, response: str,
                      out: dict = None) -> dict:
        result = {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "original_answer": sample.answer,
            "original_options": sample.options,
            "eval_options": prepared["options"],
            "prompt": prepared["prompt"],
            "response": response,
            "expected_answer": prepared.get("expected_answer"),
            "metadata": sample.metadata,
        }
        if out is not None:
            result["logprobs"] = out["logprobs"]
            result["confidence"] = out["confidence"]
            if out.get("probs") is not None:
                result["probs"] = out["probs"]
        result.update(prepared.get("extra", {}))
        return result

    # ------------------------------------------------------------------
    # Batch inference: vLLM path (continuous batching, tensor parallelism)
    # ------------------------------------------------------------------
    def _batch_run(self, samples: list, results_path: str,
                   use_logprobs: bool):
        """Run evaluation using vLLM batch inference in the main process."""
        worker_tag = f"[{self.name}|vLLM]"

        try:
            from models import build_model
            model = build_model(
                self.model_name,
                model_path=self.model_path,
                max_frames=self.max_frames,
                device=self.model_kwargs.get("device", "cuda:0"),
                **self.model_kwargs,
            )
        except Exception:
            logger.error(
                f"{worker_tag} Failed to load model:\n{traceback.format_exc()}"
            )
            return

        batch_size = getattr(model, "batch_size", 32)

        # Prepare all samples
        prepared_items = []
        valid_samples = []
        for sample in samples:
            if not os.path.exists(sample.video_path):
                logger.warning(
                    f"{worker_tag} Video not found: {sample.video_path}"
                )
                continue

            prepared = self.prepare_sample(sample)

            item = {
                "video_path": sample.video_path,
                "prompt": prepared["prompt"],
            }
            if use_logprobs:
                option_letters = [
                    chr(ord("A") + i) for i in range(len(prepared["options"]))
                ]
                item["option_tokens"] = option_letters

            prepared_items.append(item)
            valid_samples.append((sample, prepared))

        logger.info(
            f"{worker_tag} Processing {len(prepared_items)} samples "
            f"in batches of {batch_size}"
        )

        # Process in batches
        for batch_start in range(0, len(prepared_items), batch_size):
            batch_end = min(batch_start + batch_size, len(prepared_items))
            batch_items = prepared_items[batch_start:batch_end]
            batch_samples = valid_samples[batch_start:batch_end]

            logger.info(
                f"{worker_tag} Batch {batch_start // batch_size + 1}: "
                f"samples {batch_start + 1}-{batch_end}/{len(prepared_items)}"
            )

            try:
                if not use_logprobs:
                    responses = model.generate_batch(
                        batch_items,
                        max_new_tokens=self.max_new_tokens,
                    )
                    for i, (sample, prepared) in enumerate(batch_samples):
                        response = responses[i]
                        result = self._build_result(sample, prepared, response)
                        self._append_result(result, results_path)
                        logger.info(
                            f"{worker_tag} {sample.sample_id} | "
                            f"expected={result['expected_answer']} | "
                            f"response={response!r:.200}"
                        )
                else:
                    outputs = model.generate_batch_with_logprobs(
                        batch_items,
                        max_new_tokens=self.max_new_tokens,
                    )
                    for i, (sample, prepared) in enumerate(batch_samples):
                        out = outputs[i]
                        response = out["response"]
                        result = self._build_result(sample, prepared, response, out)
                        self._append_result(result, results_path)
                        logger.info(
                            f"{worker_tag} {sample.sample_id} | "
                            f"expected={result['expected_answer']} | "
                            f"response={response!r:.200}"
                        )

            except Exception as e:
                logger.error(
                    f"{worker_tag} Batch error: {e}\n{traceback.format_exc()}"
                )
                continue

        logger.info(f"{worker_tag} Batch inference complete.")

    # ------------------------------------------------------------------
    # Worker: runs on one GPU, processes its share of samples
    # ------------------------------------------------------------------
    def _worker_run(self, rank: int, samples: list,
                    results_path: str, use_logprobs: bool):
        device = f"cuda:{rank}"
        worker_tag = f"[{self.name}|GPU{rank}]"

        try:
            # Each worker loads its own model on its own GPU
            from models import build_model
            model = build_model(
                self.model_name,
                model_path=self.model_path,
                max_frames=self.max_frames,
                device=device,
                **self.model_kwargs,
            )
        except Exception:
            logger.error(
                f"{worker_tag} Failed to load model:\n{traceback.format_exc()}"
            )
            sys.exit(1)

        # Group samples by video_path to reduce video decoding overhead
        samples_by_video = self._group_samples_by_video(samples)
        logger.info(
            f"{worker_tag} Grouped {len(samples)} samples into "
            f"{len(samples_by_video)} unique videos"
        )

        desc = f"{worker_tag} ({len(samples)} samples)"
        pbar = tqdm(total=len(samples), desc=desc, dynamic_ncols=True, position=rank)
        consecutive_errors = 0

        for video_path, video_samples in samples_by_video.items():
            if not os.path.exists(video_path):
                logger.warning(f"{worker_tag} Video not found: {video_path}")
                pbar.update(len(video_samples))
                continue

            for sample in video_samples:
                prepared = self.prepare_sample(sample)

                try:
                    if not use_logprobs:
                        response = model.generate(
                            sample.video_path,
                            prepared["prompt"],
                            max_new_tokens=self.max_new_tokens,
                        )
                        out = None
                    else:
                        option_letters = [
                            chr(ord("A") + i) for i in range(len(prepared["options"]))
                        ]
                        out = model.generate_with_logprobs(
                            sample.video_path,
                            prepared["prompt"],
                            option_letters,
                        )
                        response = out["response"]

                    consecutive_errors = 0  # reset on success

                    result = self._build_result(sample, prepared, response, out)

                    # Record API token usage if available (e.g. api_models)
                    if hasattr(model, "last_usage") and model.last_usage:
                        result["usage"] = model.last_usage.copy()
                    if hasattr(model, "total_usage"):
                        result["cumulative_usage"] = model.total_usage.copy()

                    # Save immediately (append JSONL, one line per sample)
                    self._append_result(result, results_path)

                    # Log the MLLM output so user can see it in real time
                    usage_str = ""
                    if hasattr(model, "last_usage") and model.last_usage:
                        u = model.last_usage
                        usage_str = (f" | tokens: in={u.get('prompt_tokens', 0)}"
                                     f" out={u.get('completion_tokens', 0)}")
                    logger.info(
                        f"{worker_tag} {sample.sample_id} | "
                        f"expected={result['expected_answer']}{usage_str} | "
                        f"response={response!r:.200}"
                    )

                    pbar.update(1)

                except Exception as e:
                    err_msg = str(e)
                    logger.error(
                        f"{worker_tag} Error on {sample.sample_id}: {err_msg}\n"
                        f"{traceback.format_exc()}"
                    )

                    # CUDA errors are unrecoverable — abort this worker
                    if "cuda" in err_msg.lower():
                        consecutive_errors += 1
                        if consecutive_errors >= 3:
                            logger.error(
                                f"{worker_tag} 3 consecutive CUDA errors, aborting worker. "
                                f"The GPU state is likely corrupted."
                            )
                            break
                    pbar.update(1)
                    continue

        pbar.close()

        # Log cumulative token usage for API-based models
        if hasattr(model, "total_usage") and any(model.total_usage.values()):
            u = model.total_usage
            logger.info(
                f"{worker_tag} Token usage summary: "
                f"prompt={u['prompt_tokens']:,} "
                f"completion={u['completion_tokens']:,} "
                f"total={u['total_tokens']:,}"
            )

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------
    def _append_result(self, result: dict, path: str):
        """Append a single result as one JSONL line (file-lock safe for multi-process)."""
        line = json.dumps(result, ensure_ascii=False) + "\n"
        with open(path, "a") as f:
            f.write(line)

    def _load_all_results(self, path: str) -> list[dict]:
        """Load all results from JSONL file, deduplicating by sample_id."""
        if not os.path.exists(path):
            return []
        seen = set()
        results = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["sample_id"] not in seen:
                    seen.add(r["sample_id"])
                    results.append(r)
        return results

    @staticmethod
    def _split_samples(samples: list, n: int) -> list[list]:
        """Split samples into n roughly equal chunks (round-robin)."""
        chunks = [[] for _ in range(n)]
        for i, s in enumerate(samples):
            chunks[i % n].append(s)
        return chunks

    @staticmethod
    def _group_samples_by_video(samples: list) -> dict:
        """Group samples by video_path to reduce video decoding overhead.

        Returns:
            Dict mapping video_path -> list[MCQSample]
        """
        from collections import defaultdict
        grouped = defaultdict(list)
        for sample in samples:
            grouped[sample.video_path].append(sample)
        return dict(grouped)
