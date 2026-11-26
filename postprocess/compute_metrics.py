"""
Compute the paper metrics (ACC / MCDR / OEDR / UDR) from raw inference results.

Input layout (``main.py`` output, e.g. ``outputs/`` or the released
``inference_results/``; files may be ``.jsonl`` or ``.jsonl.gz``), optionally
with ``extract_answer_llm.py`` outputs next to it::

    <root>/<benchmark>/<model>/<eval_method>/*.jsonl[.gz]
    <root>/<benchmark>/<model>/<eval_method>_postprocessed/*.jsonl[.gz]   # optional
    <root>/<benchmark>/<model>_postprocessed/<eval_method>/*.jsonl[.gz]   # optional

When both a raw and a ``*_postprocessed`` version of the same
(benchmark, model, eval_method) exist, the postprocessed one is used. If
several files of the same kind exist (e.g. across multiple ``--input-dirs``),
the one from the earlier root is used.

Metrics per eval_method:
  - baseline[_cot]                  -> ACC : accuracy (predicted == expected)
  - remove_correct_add_none[_cot]   -> MCDR: proportion selecting the "None of the above" option
  - remove_correct_force_none[_cot] -> OEDR: proportion outputting "None of the above" text
  - remove_correct[_cot]            -> UDR : proportion NOT selecting any valid option

Answer parsing: if a record has a non-null ``predicted_answer`` (only written
by ``extract_answer_llm.py``; raw inference results have none) it is used;
otherwise the letter is parsed from ``response`` with regular expressions.

Outputs:
    <eval_dir>/<benchmark>/<model>/<eval_method>/eval_result.json
    <eval_dir>/summary.json

Each eval_result.json also contains a per-``task_type`` breakdown (used for
the Video-MME temporal subsets, see ``temporal_breakdown.py``).

Usage:
    python postprocess/compute_metrics.py --input-dirs inference_results --eval-dir eval_results
"""

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
POSTPROCESSED_SUFFIX = "_postprocessed"


# ---------------------------------------------------------------------------
# Answer extraction (adapted from utils/prompt_utils.py)
# ---------------------------------------------------------------------------

def parse_answer_letter(response: str, valid_letters: set[str]) -> str | None:
    """Extract the answer letter from a model response using regex.

    Args:
        response: Raw model response text.
        valid_letters: Set of valid option letters (e.g. {'A','B','C','D'}).

    Returns:
        Single uppercase letter, or None if no valid answer found.
    """
    if not response:
        return None

    response = response.strip()

    # Pattern 1: Explicit answer markers — "Answer: A", "The answer is A"
    m = re.search(
        r"(?:answer|choice|option)\s*(?:is|:)\s*\(?([A-Z])\)?",
        response, re.IGNORECASE,
    )
    if m and m.group(1).upper() in valid_letters:
        return m.group(1).upper()

    # Pattern 2: Starts with a letter followed by punctuation/space
    m = re.match(r"^\(?([A-Z])\)?[\.\)\:\s,]", response, re.IGNORECASE)
    if m and m.group(1).upper() in valid_letters:
        return m.group(1).upper()

    # Pattern 3: Just a single letter
    if len(response) == 1 and response.upper() in valid_letters:
        return response.upper()

    # Pattern 4: First standalone valid letter
    for m in re.finditer(r"\b([A-Z])\b", response):
        if m.group(1).upper() in valid_letters:
            return m.group(1).upper()

    return None


def get_predicted_answer(rec: dict, valid_letters: set[str]) -> str | None:
    """Get the predicted answer from a record.

    Uses the pre-extracted 'predicted_answer' field if available (e.g. from
    LLM-based extraction in extract_answer_llm.py), otherwise falls back
    to regex-based parsing of the 'response' field.
    """
    # Use pre-extracted answer if present
    if "predicted_answer" in rec and rec["predicted_answer"] is not None:
        ans = rec["predicted_answer"].upper()
        if ans in valid_letters:
            return ans
    # Fallback to regex
    return parse_answer_letter(rec.get("response", ""), valid_letters)


def response_contains_none_of_above(response: str) -> bool:
    """Check if response text contains 'None of the above' (case-insensitive)."""
    return bool(re.search(r"none\s+of\s+the\s+above", response, re.IGNORECASE))


# ---------------------------------------------------------------------------
# Per-setting evaluation
# ---------------------------------------------------------------------------

def eval_baseline(records: list[dict]) -> dict:
    """ACC: predicted == expected."""
    total = len(records)
    correct = 0
    answer_dist = Counter()
    null_count = 0
    all_max_confs = []

    for rec in records:
        valid = set(OPTION_LETTERS[:rec["num_options"]])
        predicted = get_predicted_answer(rec, valid)
        if predicted is None:
            null_count += 1
            answer_dist["NULL"] += 1
        else:
            answer_dist[predicted] += 1
            if predicted == rec["expected_answer"]:
                correct += 1

        # Collect confidence statistics
        probs = rec.get("probs")
        if probs:
            max_conf = max(probs.values())
            all_max_confs.append(max_conf)

    result = {
        "metric": "accuracy",
        "value": correct / total if total else 0,
        "total": total,
        "correct": correct,
        "null_count": null_count,
        "answer_distribution": dict(sorted(answer_dist.items())),
    }
    if all_max_confs:
        result["max_confidences"] = all_max_confs
    return result


def eval_remove_correct_add_none(records: list[dict]) -> dict:
    """MCDR: proportion selecting the 'None of the above' option letter."""
    total = len(records)
    none_count = 0
    answer_dist = Counter()
    null_count = 0
    none_confidences = []
    max_confidences = []

    for rec in records:
        valid = set(OPTION_LETTERS[:rec["num_options"]])
        predicted = get_predicted_answer(rec, valid)

        if predicted is None:
            # Check if response literally says "None of the above"
            if response_contains_none_of_above(rec["response"]):
                # Map to the None option letter (last option)
                none_letter = OPTION_LETTERS[rec["num_options"] - 1]
                predicted = none_letter

        if predicted is None:
            null_count += 1
            answer_dist["NULL"] += 1
        else:
            answer_dist[predicted] += 1
            # The "None of the above" is always the last option
            none_letter = OPTION_LETTERS[rec["num_options"] - 1]
            if predicted == none_letter:
                none_count += 1

        # Collect confidence values for plotting
        probs = rec.get("probs")
        if probs:
            none_label = rec.get("none_label", OPTION_LETTERS[rec["num_options"] - 1])
            none_confidences.append(probs.get(none_label, 0.0))
            max_confidences.append(max(probs.values()))

    result = {
        "metric": "none_of_above_rate",
        "value": none_count / total if total else 0,
        "total": total,
        "none_count": none_count,
        "null_count": null_count,
        "answer_distribution": dict(sorted(answer_dist.items())),
    }
    if none_confidences:
        result["none_confidences"] = none_confidences
        result["max_confidences"] = max_confidences
    return result


def eval_remove_correct_force_none(records: list[dict]) -> dict:
    """OEDR: proportion of responses that output 'None of the above' text."""
    total = len(records)
    none_count = 0
    answer_dist = Counter()
    null_count = 0

    for rec in records:
        valid = set(OPTION_LETTERS[:rec["num_options"]])
        predicted = get_predicted_answer(rec, valid)
        is_none = response_contains_none_of_above(rec["response"])

        if is_none:
            none_count += 1
            answer_dist["NONE_OF_ABOVE"] += 1
        elif predicted is not None:
            answer_dist[predicted] += 1
        else:
            null_count += 1
            answer_dist["NULL"] += 1

    return {
        "metric": "none_of_above_output_rate",
        "value": none_count / total if total else 0,
        "total": total,
        "none_count": none_count,
        "null_count": null_count,
        "answer_distribution": dict(sorted(answer_dist.items())),
    }


def eval_remove_correct(records: list[dict]) -> dict:
    """UDR: proportion NOT selecting any valid option letter."""
    total = len(records)
    no_selection_count = 0
    answer_dist = Counter()

    for rec in records:
        valid = set(OPTION_LETTERS[:rec["num_options"]])
        predicted = get_predicted_answer(rec, valid)

        if predicted is None:
            no_selection_count += 1
            answer_dist["NO_VALID_OPTION"] += 1
        else:
            answer_dist[predicted] += 1

    return {
        "metric": "no_selection_rate",
        "value": no_selection_count / total if total else 0,
        "total": total,
        "no_selection_count": no_selection_count,
        "answer_distribution": dict(sorted(answer_dist.items())),
    }


# eval_method -> (eval function, paper metric name)
EVAL_FNS = {
    "baseline": (eval_baseline, "ACC"),
    "baseline_cot": (eval_baseline, "ACC"),
    "remove_correct": (eval_remove_correct, "UDR"),
    "remove_correct_cot": (eval_remove_correct, "UDR"),
    "remove_correct_add_none": (eval_remove_correct_add_none, "MCDR"),
    "remove_correct_add_none_cot": (eval_remove_correct_add_none, "MCDR"),
    "remove_correct_force_none": (eval_remove_correct_force_none, "OEDR"),
    "remove_correct_force_none_cot": (eval_remove_correct_force_none, "OEDR"),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def iter_result_files(root: Path):
    """All result files under root (``*.jsonl`` and ``*.jsonl.gz``), sorted."""
    return sorted(list(root.rglob("*.jsonl")) + list(root.rglob("*.jsonl.gz")))


def load_records(path: Path) -> list[dict]:
    records = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _strip_suffix(name: str) -> tuple[str, bool]:
    if name.endswith(POSTPROCESSED_SUFFIX):
        return name[: -len(POSTPROCESSED_SUFFIX)], True
    return name, False


def collect_tasks(roots: list[Path]) -> list[dict]:
    """Collect one task per (benchmark, model, eval_method).

    Expected layout: root/<benchmark>/<model>/<eval_method>/*.jsonl, where the
    model or eval_method directory may carry a ``_postprocessed`` suffix.
    Postprocessed results take precedence over raw ones.
    """
    tasks: dict[tuple, dict] = {}
    for root in roots:
        for fpath in iter_result_files(root):
            parts = fpath.relative_to(root).parts
            if len(parts) < 4:
                continue
            dataset = parts[0]
            model, pp_model = _strip_suffix(parts[1])
            experiment, pp_exp = _strip_suffix(parts[2])
            postprocessed = pp_model or pp_exp

            if experiment not in EVAL_FNS:
                print(f"SKIP unknown eval_method '{experiment}': {fpath}")
                continue

            key = (dataset, model, experiment)
            prev = tasks.get(key)
            if prev is not None:
                if prev["postprocessed"] and not postprocessed:
                    continue  # keep the postprocessed version
                if prev["postprocessed"] == postprocessed:
                    # Same kind: the earlier root / file wins.
                    print(f"WARNING: multiple result files for {'/'.join(key)}; "
                          f"using {prev['path']}, ignoring {fpath}")
                    continue
            tasks[key] = {
                "path": fpath,
                "dataset": dataset,
                "model": model,
                "experiment": experiment,
                "postprocessed": postprocessed,
            }
    return [tasks[k] for k in sorted(tasks)]


def main():
    parser = argparse.ArgumentParser(
        description="Compute ACC / MCDR / OEDR / UDR from result JSONL files."
    )
    parser.add_argument(
        "--input-dirs", nargs="+", required=True,
        help="Root directories laid out as <benchmark>/<model>/<eval_method>/*.jsonl[.gz], "
             "e.g. inference_results/ (paper results) or outputs/ (your own runs)."
    )
    parser.add_argument(
        "--eval-dir", type=str, default="eval_results",
        help="Output directory for evaluation results. Default: eval_results/"
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)

    roots = []
    for d in args.input_dirs:
        p = Path(d)
        if not p.exists():
            print(f"WARNING: {p} does not exist, skipping.")
            continue
        roots.append(p)
    if not roots:
        print("ERROR: no valid input directories.")
        sys.exit(1)

    all_tasks = collect_tasks(roots)
    print(f"Found {len(all_tasks)} result files to evaluate\n")

    eval_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for task in all_tasks:
        fpath = task["path"]
        dataset = task["dataset"]
        model = task["model"]
        experiment = task["experiment"]
        eval_fn, paper_metric = EVAL_FNS[experiment]

        records = load_records(fpath)
        n_unique = len({r.get("sample_id") for r in records})
        if n_unique != len(records):
            print(f"WARNING: {fpath} has {len(records)} records but only {n_unique} "
                  f"unique sample_ids; duplicates are counted as-is.")
        result = eval_fn(records)
        result["paper_metric"] = paper_metric
        result["dataset"] = dataset
        result["model"] = model
        result["experiment"] = experiment
        result["postprocessed"] = task["postprocessed"]
        result["source"] = str(fpath)

        # Add per-task-type breakdowns (using metadata.task_type)
        by_task_type = {}
        task_groups: dict[str, list[dict]] = {}
        for rec in records:
            tt = rec.get("metadata", {}).get("task_type", "unknown")
            task_groups.setdefault(tt, []).append(rec)
        for tt, group in sorted(task_groups.items()):
            by_task_type[tt] = eval_fn(group)
        result["task_type"] = by_task_type

        key = f"{dataset}/{model}/{experiment}"
        all_results[key] = result

        # Save per-file eval result
        out_path = eval_dir / dataset / model / experiment / "eval_result.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        # Print summary
        dist_str = "  ".join(f"{k}:{v}" for k, v in sorted(result["answer_distribution"].items()))
        tag = " [postprocessed]" if task["postprocessed"] else ""
        print(f"{key}{tag}")
        print(f"  {paper_metric} ({result['metric']}) = {result['value']:.4f}  "
              f"(n={result['total']})  [{dist_str}]")
        if result.get("null_count", 0) > 0:
            print(f"  ** {result['null_count']} NULL answers")

    # Save summary JSON
    summary_path = eval_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    # Print summary table grouped by eval_method
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    for experiment, (eval_fn, paper_metric) in EVAL_FNS.items():
        matching = {k: v for k, v in all_results.items() if k.endswith(f"/{experiment}")}
        if not matching:
            continue
        print(f"\n--- {experiment} | {paper_metric} ({eval_fn.__doc__.strip()}) ---")
        print(f"{'dataset/model':<55} {paper_metric:>8}  {'n':>5}  {'nulls':>5}")
        for key in sorted(matching):
            r = matching[key]
            null_display = r.get("null_count", "-")
            print(f"  {key:<53} {100 * r['value']:>8.1f}  {r['total']:>5}  {null_display:>5}")


if __name__ == "__main__":
    main()
