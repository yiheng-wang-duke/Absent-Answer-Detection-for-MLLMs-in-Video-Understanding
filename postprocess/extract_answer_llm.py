"""
LLM-based answer extraction for long / free-form responses.

The paper (Sec. 3.2) uses Qwen3-1.7B to extract the selected option (or an
explicit "none of the options is correct") from long responses: all CoT
settings and the API models (Gemini-2.5-Flash/Pro, GPT-4o). Short responses
that reduce to an option letter do not need this step.

For each input directory ``<dir>``, every ``*.jsonl`` / ``*.jsonl.gz`` inside is
processed and written (same file name and format) to the sibling directory
``<dir>_postprocessed`` with
``predicted_answer`` (a letter, or null for "None") and ``correct`` filled in.
``compute_metrics.py`` automatically prefers ``*_postprocessed`` results over
the raw ones.

Supports multi-GPU inference: one model per GPU, files sharded across GPUs.

Usage:
    python postprocess/extract_answer_llm.py \
        --input-dirs inference_results/videomme/qwen3_vl_64frms/remove_correct_add_none_cot \
                     inference_results/videomme/qwen3_vl_64frms/remove_correct_force_none_cot \
        --batch-size 32 --num-gpus 1
"""

import argparse
import gzip
import json
import sys
import os
from pathlib import Path
from multiprocessing import Process, Queue

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-1.7B"
OPTION_LETTERS = "ABCDE"

def is_baseline_experiment(path) -> bool:
    """Whether a JSONL path belongs to a baseline-style experiment.

    Baseline / baseline_cot options do NOT include a
    "None of the above" choice, so extraction should only match a letter and
    never treat the response as an abstention.
    """
    name = str(path).lower()
    # remove_correct_* experiments legitimately have a None option; keep those.
    if "remove_correct" in name:
        return False
    return ("baseline" in name) or ("non_visual" in name)


def build_prompt(
    response: str, options: list[str], num_options: int, allow_none: bool = True
) -> str:
    """Build a prompt asking the LLM to extract the chosen answer.

    When ``allow_none`` is False (e.g. baseline / baseline_cot, where the
    options do not include a "None of the above" choice), the model is asked
    to reply with an option letter only — no None / abstention option.
    """
    option_str = "\n".join(
        f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options[:num_options])
    )
    letters = ", ".join(OPTION_LETTERS[:num_options])
    if allow_none:
        instruction = (
            f"Which option did the model choose? "
            f"Reply with ONLY the letter ({letters}) "
            f"or \"None\" if the model explicitly states none of the options are correct "
            f"or refuses to answer. Do not explain."
        )
    else:
        instruction = (
            f"Which option did the model choose? "
            f"Reply with ONLY the letter ({letters}). "
            f"If the response is ambiguous, pick the single most likely option letter. "
            f"Do not explain."
        )
    return (
        f"Given the following multiple-choice options:\n{option_str}\n\n"
        f"The model produced this response:\n\"\"\"\n{response}\n\"\"\"\n\n"
        f"{instruction}"
    )


def extract_answer(
    generated_text: str, num_options: int, allow_none: bool = True
) -> str | None:
    """Parse the LLM's short extraction output into a letter or None.

    When ``allow_none`` is False, a "NONE" output is not treated as an
    abstention: only a valid option letter (A..) is returned.
    """
    text = generated_text.strip().upper()
    # Remove </think>... wrapper if present (Qwen3 thinking mode)
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    # Take only the first line
    text = text.split("\n")[0].strip()
    valid = set(OPTION_LETTERS[:num_options])
    # Check for single letter
    if len(text) == 1 and text in valid:
        return text
    # Check starts with a letter
    if text and text[0] in valid:
        return text[0]
    # "NONE" -> None (only when abstention is allowed)
    if allow_none and "NONE" in text:
        return None
    # Last resort: find first valid letter
    for ch in text:
        if ch in valid:
            return ch
    return None


def worker_fn(gpu_id: int, tasks: list[dict], batch_size: int, result_queue: Queue,
              model_name: str = MODEL_NAME):
    """
    Worker process: loads model on a specific GPU and processes assigned tasks.
    Each task is a dict with keys: jsonl_path, out_path, records, prompts.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    print(f"[GPU {gpu_id}] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    print(f"[GPU {gpu_id}] Model loaded. Processing {len(tasks)} file(s).")

    for task in tasks:
        jsonl_path = task["jsonl_path"]
        out_path = task["out_path"]
        records = task["records"]
        prompts = task["prompts"]
        allow_none = task.get("allow_none", True)

        print(f"[GPU {gpu_id}] Processing: {jsonl_path} ({len(records)} records)")

        predicted_answers = []
        pbar = tqdm(
            total=len(prompts),
            desc=f"[GPU {gpu_id}] {Path(jsonl_path).parent.name}",
            unit="rec",
            file=sys.stdout,
        )

        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

            # Format as chat messages
            batch_messages = [
                [{"role": "user", "content": p}] for p in batch_prompts
            ]
            batch_texts = [
                tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
                for msgs in batch_messages
            ]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            # Decode only the new tokens
            for i, output_ids in enumerate(outputs):
                idx = batch_start + i
                input_len = inputs["input_ids"][i].shape[0]
                new_tokens = output_ids[input_len:]
                generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
                num_opts = records[idx]["num_options"]
                answer = extract_answer(generated, num_opts, allow_none=allow_none)
                predicted_answers.append(answer)

            pbar.update(batch_end - batch_start)
        pbar.close()

        # Update records with new predictions
        changed = 0
        for rec, new_ans in zip(records, predicted_answers):
            old_ans = rec.get("predicted_answer")
            rec["predicted_answer"] = new_ans
            if new_ans is not None:
                rec["correct"] = (new_ans == rec["expected_answer"])
            else:
                rec["correct"] = False
            if old_ans != new_ans:
                changed += 1

        null_count = sum(1 for a in predicted_answers if a is None)
        print(f"[GPU {gpu_id}] {Path(jsonl_path).name}: changed={changed}/{len(records)}, nulls={null_count}")

        # Write output (same format as the input: .jsonl or .jsonl.gz)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if str(out_path).endswith(".gz") else open
        with opener(out_path, "wt") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[GPU {gpu_id}] Saved: {out_path}")

        result_queue.put({
            "gpu_id": gpu_id,
            "jsonl_path": str(jsonl_path),
            "out_path": str(out_path),
            "total": len(records),
            "changed": changed,
            "nulls": null_count,
        })


def main():
    parser = argparse.ArgumentParser(
        description="Extract answer letters from model responses using LLM."
    )
    parser.add_argument(
        "--input-dirs", nargs="+", required=True,
        help="One or more input directories containing JSONL files (searched recursively)."
    )
    parser.add_argument(
        "--output-suffix", type=str, default="_postprocessed",
        help="Suffix appended to each input directory name for output. Default: _postprocessed"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--model", type=str, default=MODEL_NAME,
        help=f"Extractor model. Default (paper): {MODEL_NAME}"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=None,
        help="Number of GPUs to use. Default: all available GPUs."
    )
    parser.add_argument("--dry-run", action="store_true", help="Only show prompts, don't run model")
    args = parser.parse_args()

    # Determine number of GPUs
    if args.num_gpus is not None:
        num_gpus = args.num_gpus
    else:
        num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        print("ERROR: No GPUs available.")
        sys.exit(1)
    print(f"Using {num_gpus} GPU(s)")

    # Collect all JSONL files and build tasks
    all_tasks = []
    for input_dir in args.input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"WARNING: {input_path} does not exist, skipping.")
            continue

        # Output directory: append suffix to the input directory name
        out_path_base = input_path.parent / (input_path.name + args.output_suffix)

        jsonl_files = sorted(list(input_path.rglob("*.jsonl")) + list(input_path.rglob("*.jsonl.gz")))
        print(f"Input: {input_path} -> {len(jsonl_files)} JSONL file(s)")
        print(f"Output: {out_path_base}")

        for jsonl_path in jsonl_files:
            # Compute relative path for output
            rel_path = jsonl_path.relative_to(input_path)
            out_file = out_path_base / rel_path

            # Read records
            records = []
            opener = gzip.open if jsonl_path.suffix == ".gz" else open
            with opener(jsonl_path, "rt") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

            if not records:
                print(f"  SKIP (empty): {jsonl_path}")
                continue

            # Baseline-style experiments have no "None of the above" option:
            # extract a letter only, no abstention.
            allow_none = not is_baseline_experiment(jsonl_path)
            if not allow_none:
                print(f"  (baseline-style: letters only, no None) {jsonl_path.name}")

            # Build prompts
            prompts = []
            for rec in records:
                prompts.append(build_prompt(
                    rec["response"],
                    rec["eval_options"],
                    rec["num_options"],
                    allow_none=allow_none,
                ))

            all_tasks.append({
                "jsonl_path": str(jsonl_path),
                "out_path": str(out_file),
                "records": records,
                "prompts": prompts,
                "allow_none": allow_none,
            })

    print(f"\nTotal tasks (JSONL files): {len(all_tasks)}")
    total_records = sum(len(t["records"]) for t in all_tasks)
    print(f"Total records: {total_records}")

    if args.dry_run:
        print("\n[DRY RUN] Sample prompt from first task:")
        if all_tasks:
            print(all_tasks[0]["prompts"][0][:500])
        return

    if not all_tasks:
        print("No tasks to process.")
        return

    # Distribute tasks across GPUs (round-robin by file)
    gpu_tasks = [[] for _ in range(num_gpus)]
    for i, task in enumerate(all_tasks):
        gpu_tasks[i % num_gpus].append(task)

    for g in range(num_gpus):
        n_files = len(gpu_tasks[g])
        n_recs = sum(len(t["records"]) for t in gpu_tasks[g])
        print(f"  GPU {g}: {n_files} file(s), {n_recs} record(s)")

    # Launch worker processes
    result_queue = Queue()
    processes = []
    for gpu_id in range(num_gpus):
        if not gpu_tasks[gpu_id]:
            continue
        p = Process(
            target=worker_fn,
            args=(gpu_id, gpu_tasks[gpu_id], args.batch_size, result_queue, args.model),
        )
        p.start()
        processes.append(p)

    # Wait for all processes
    for p in processes:
        p.join()

    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    print(f"\n{'='*60}")
    print(f"All done. Processed {len(results)} file(s).")
    total_changed = sum(r["changed"] for r in results)
    total_nulls = sum(r["nulls"] for r in results)
    print(f"Total changed: {total_changed}/{total_records}")
    print(f"Total nulls: {total_nulls}/{total_records}")


if __name__ == "__main__":
    main()
