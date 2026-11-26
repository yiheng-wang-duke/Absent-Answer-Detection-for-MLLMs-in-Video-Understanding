"""
Table 3: ACC / MCDR / OEDR / UDR on the Video-MME temporal subsets.

Reads the per-task_type breakdown written by ``compute_metrics.py`` and prints
one table per subset ("Temporal Perception": 55 questions,
"Temporal Reasoning": 177 questions).

Usage:
    python postprocess/temporal_breakdown.py --eval-dir eval_results
"""

import argparse
import collections
import json
from pathlib import Path

TARGET_TASKS = ["Temporal Perception", "Temporal Reasoning"]
# Column order follows Table 3 of the paper.
EXPERIMENTS = [
    ("baseline", "ACC"),
    ("remove_correct_add_none", "MCDR"),
    ("remove_correct_force_none", "OEDR"),
    ("remove_correct", "UDR"),
]


def main():
    parser = argparse.ArgumentParser(description="Video-MME temporal-subset breakdown (Table 3).")
    parser.add_argument("--eval-dir", type=str, default="eval_results",
                        help="Output directory of compute_metrics.py. Default: eval_results/")
    args = parser.parse_args()

    videomme_dir = Path(args.eval_dir) / "videomme"
    if not videomme_dir.exists():
        print(f"ERROR: {videomme_dir} does not exist. Run compute_metrics.py first.")
        return

    # model -> task -> exp -> (value, total)
    data = collections.defaultdict(lambda: collections.defaultdict(dict))

    for model_dir in sorted(videomme_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for exp, _ in EXPERIMENTS:
            result_path = model_dir / exp / "eval_result.json"
            if not result_path.exists():
                continue
            with open(result_path) as f:
                result = json.load(f)
            by_task_type = result.get("task_type", {})
            for task in TARGET_TASKS:
                tt = by_task_type.get(task)
                if tt is None or tt.get("total", 0) == 0:
                    continue
                data[model_dir.name][task][exp] = (tt["value"], tt["total"])

    for task in TARGET_TASKS:
        print(f"\n{'=' * 70}")
        print(f"Video-MME - {task}")
        print(f"{'=' * 70}")
        print(f"{'model':<30}" + "".join(f"{name:>10}" for _, name in EXPERIMENTS) + f"{'n':>6}")
        print("-" * 76)
        for model in sorted(data):
            exps = data[model].get(task, {})
            if not exps:
                continue
            row = f"{model:<30}"
            n = None
            for exp, _ in EXPERIMENTS:
                if exp in exps:
                    value, n = exps[exp]
                    row += f"{100 * value:>10.1f}"
                else:
                    row += f"{'--':>10}"
            print(row + f"{n if n is not None else '':>6}")


if __name__ == "__main__":
    main()
