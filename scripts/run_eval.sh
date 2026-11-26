#!/usr/bin/env bash
# Run all evaluation settings for one model on one benchmark.
#
# Usage:
#   bash scripts/run_eval.sh <benchmark> <model> <video_dir> [num_gpus] [max_frames]
#
# Example:
#   bash scripts/run_eval.sh videomme  qwen3_vl /data/Video-MME/data 2 64
#   bash scripts/run_eval.sh egoschema qwen3_vl /data/egoschema/videos 2 64

set -euo pipefail

BENCHMARK=${1:?benchmark (videomme | egoschema)}
MODEL=${2:?model name (see main.py)}
VIDEO_DIR=${3:?path to video directory}
NUM_GPUS=${4:-1}
MAX_FRAMES=${5:-64}   # paper default (Gemma3-12B on Video-MME uses 32)

# Paper settings (Section 3.1) -> eval_method
#   ACC  : baseline
#   UDR  : remove_correct
#   MCDR : remove_correct_add_none
#   OEDR : remove_correct_force_none
# CoT variants (Section 5, Table 4): append _cot
EVAL_METHODS=(
    baseline
    remove_correct
    remove_correct_add_none
    remove_correct_force_none
    baseline_cot
    remove_correct_cot
    remove_correct_add_none_cot
    remove_correct_force_none_cot
)

for EVAL_METHOD in "${EVAL_METHODS[@]}"; do
    # CoT settings need a longer generation budget
    if [[ $EVAL_METHOD == *_cot ]]; then MAX_NEW_TOKENS=2048; else MAX_NEW_TOKENS=128; fi

    echo ">>> $BENCHMARK | $MODEL | $EVAL_METHOD | frames=$MAX_FRAMES"
    python main.py \
        --benchmark "$BENCHMARK" \
        --model "$MODEL" \
        --eval_method "$EVAL_METHOD" \
        --video_dir "$VIDEO_DIR" \
        --max_frames "$MAX_FRAMES" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --use_logprobs \
        --num_gpus "$NUM_GPUS" \
        --output_dir "outputs/${BENCHMARK}/${MODEL}_${MAX_FRAMES}frms/${EVAL_METHOD}"
done
