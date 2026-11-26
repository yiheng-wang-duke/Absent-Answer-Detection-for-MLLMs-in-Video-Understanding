#!/usr/bin/env bash
# Compute the paper metrics from raw inference results.
#
#   <results_dir>/<benchmark>/<model>/<eval_method>/*.jsonl[.gz]   (raw inference output)
#     -> .../<eval_method>_postprocessed/       step 1: Qwen3-1.7B answer extraction (long responses only)
#     -> <eval_dir>/...                          step 2: ACC / MCDR / OEDR / UDR (+ Table 3)
#
# Usage:
#   bash scripts/postprocess.sh [results_dir] [eval_dir]
#
#   results_dir: inference_results (released paper results, default) or outputs (your own runs)

set -euo pipefail

RESULTS=${1:-inference_results}
EVAL_DIR=${2:-eval_results}

# Step 1: LLM answer extraction for long / free-form responses:
# all *_cot settings and the API models (Gemini, GPT-4o). Requires a GPU.
EXTRACT_DIRS=()
for d in "$RESULTS"/*/*/*_cot "$RESULTS"/*/gemini-*/* "$RESULTS"/*/gpt-*/*; do
    [[ -d $d && $d != *_postprocessed ]] && EXTRACT_DIRS+=("$d")
done
if (( ${#EXTRACT_DIRS[@]} )); then
    python postprocess/extract_answer_llm.py --input-dirs "${EXTRACT_DIRS[@]}" \
        --batch-size 32 --num-gpus 1
fi

# Step 2: metrics (postprocessed results take precedence over raw ones).
python postprocess/compute_metrics.py --input-dirs "$RESULTS" --eval-dir "$EVAL_DIR"

# Table 3: Video-MME temporal subsets.
python postprocess/temporal_breakdown.py --eval-dir "$EVAL_DIR"
