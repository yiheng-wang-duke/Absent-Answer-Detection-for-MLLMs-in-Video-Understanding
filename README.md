# When No Answer Is Correct: Diagnosing Absent Answer Detection for MLLMs in Video Understanding

[![Paper](https://img.shields.io/badge/EMNLP%202026-Paper-blue)](https://aclanthology.org/2026.emnlp-main.980/)

Official code and inference results for our EMNLP 2026 paper.

## Motivation

Video understanding benchmarks share one assumption: the correct answer is always among the candidates. A model can therefore score well by picking the most plausible option, without really understanding the video. We probe this by **removing the ground-truth answer** from each multiple-choice question and keeping the video and question unchanged. A model that understands the video should recognize that **no option is correct** instead of choosing a distractor. We call this *absent answer detection*.

## Experimental Setup

Given a video, a question and its candidate set $C$ with ground truth $c^\*$, we evaluate:

| Setting | Candidates | Detection cue | Metric |
|---|---|---|---|
| Baseline | $C$ | – | **ACC**: accuracy |
| Multi-choice detection | $C \setminus \{c^\*\}$ + "None of the above" | NOTA option | **MCDR**: rate of selecting NOTA |
| Open-ended detection | $C \setminus \{c^\*\}$ | Instruction: answer "None of the above" if no option is correct | **OEDR**: rate of answering "None of the above" |
| Unprompted detection | $C \setminus \{c^\*\}$ | – (standard prompt) | **UDR**: rate of rejecting all options |

All settings are also run with a **chain-of-thought** (CoT) prompt as a mitigation (Sec. 5).

- **Benchmarks:** Video-MME (2700 QA) and the EgoSchema validation subset (500 QA).
- **Models:** Gemini-2.5-Flash/Pro, GPT-4o, Gemma3-4B/12B, Qwen2-VL-7B, Qwen2.5-VL-7B, Qwen2.5-Omni-7B, Qwen3-VL-8B, Qwen3-Omni-30B-A3B, MiMo-VL-7B, InternVL3-8B, InternVL3.5-8B.
- **Inference:** 64 frames by default and greedy decoding. Long free-form responses (CoT and API models) are parsed with Qwen3-1.7B as an answer extractor.

## Results

**Video-MME** († 32 frames)

| Model | ACC | MCDR | OEDR | UDR |
|---|---:|---:|---:|---:|
| Gemini-2.5-Flash | 68.9 | 33.9 | 43.6 | 2.4 |
| Gemma3-4B | 52.1 | 7.9 | 6.0 | 0.0 |
| Gemma3-12B† | 60.5 | 25.4 | 11.8 | 0.1 |
| Qwen2-VL-7B | 61.3 | 13.1 | 1.2 | 0.0 |
| Qwen2.5-VL-7B | 62.8 | 11.7 | 11.9 | 0.0 |
| Qwen2.5-Omni-7B | 63.7 | 19.9 | 61.7 | 0.0 |
| Qwen3-VL-8B | 67.0 | 17.4 | 16.2 | 0.7 |
| Qwen3-Omni-30B | 68.1 | 23.0 | 9.3 | 1.9 |
| MiMo-VL-7B | 64.3 | 40.9 | 8.8 | 0.0 |
| InternVL3-8B | 66.4 | 11.9 | 1.1 | 0.0 |
| InternVL3.5-8B | 65.3 | 13.6 | 6.5 | 0.0 |

**EgoSchema**

| Model | ACC | MCDR | OEDR | UDR |
|---|---:|---:|---:|---:|
| Gemini-2.5-Flash | 68.9 | 27.5 | 34.6 | 0.4 |
| Gemini-2.5-Pro | 72.0 | 60.6 | 52.6 | 1.4 |
| GPT-4o | 65.6 | 31.0 | 28.0 | 1.0 |
| Qwen2-VL-7B | 66.4 | 3.2 | 1.6 | 0.0 |
| Qwen2.5-VL-7B | 66.0 | 8.8 | 9.0 | 0.0 |
| Qwen3-VL-8B | 74.6 | 21.0 | 23.2 | 1.6 |
| Qwen3-Omni-30B | 66.4 | 26.6 | 2.8 | 0.2 |
| MiMo-VL-7B | 68.0 | 40.6 | 17.4 | 0.0 |
| InternVL3-8B | 79.2 | 9.6 | 2.2 | 0.0 |
| InternVL3.5-8B | 74.8 | 11.0 | 5.8 | 0.0 |

**Chain-of-thought on Video-MME**

| Model | ACC | MCDR | OEDR | UDR |
|---|---:|---:|---:|---:|
| InternVL3.5-8B | 65.3 | 13.6 | 6.5 | 0.0 |
| InternVL3.5-8B + CoT | 63.1 | 25.7 | 18.6 | 0.0 |
| Qwen3-VL-8B | 67.0 | 17.4 | 16.2 | 0.7 |
| Qwen3-VL-8B + CoT | 63.9 | 48.2 | 49.9 | 4.4 |

**Key findings**
- MLLMs overwhelmingly select plausible distractors instead of detecting the absent answer. Without an explicit cue, UDR is near 0%.
- Detection drops further on temporal tasks, whose distractors are semantically closer to the removed answer.
- Denser frame sampling raises ACC but *lowers* MCDR and OEDR.
- CoT substantially improves detection, but rates stay low, ACC drops slightly, and responses become much longer.

## Code

### Installation

Tested with Python 3.10, `torch==2.10.0`, `transformers==4.57.6`, `vllm==0.18.1`, CUDA 12.8.

```bash
pip install -r requirements.txt
```

Annotations are downloaded from the Hugging Face Hub (`lmms-lab/Video-MME`, `lmms-lab/egoschema`). Download the videos yourself and pass the folder with `--video_dir`. Files are expected as `{videoID}.mp4` for Video-MME and `{video_idx}.mp4` for EgoSchema.

### Reproducing the results

**From the released inference results.** No videos are needed. [inference_results/](inference_results/) holds the raw model outputs behind the paper's tables and Figure 2, as gzipped JSONL laid out as `<benchmark>/<model>_<frames>frms/<eval_method>/`. To compute all metrics:

```bash
bash scripts/postprocess.sh inference_results eval_results
```

This runs the Qwen3-1.7B answer extraction on the CoT and API-model results (needs a GPU), computes ACC / MCDR / OEDR / UDR into `eval_results/`, and prints the Video-MME temporal-subset breakdown (Table 3).

**Running inference yourself.** [scripts/run_eval.sh](scripts/run_eval.sh) runs all eight settings for one model, and post-processing works the same way:

```bash
bash scripts/run_eval.sh videomme  qwen3_vl /path/to/Video-MME/data   2 64   # <benchmark> <model> <video_dir> <num_gpus> <frames>
bash scripts/run_eval.sh egoschema qwen3_vl /path/to/egoschema/videos 2 64
bash scripts/postprocess.sh outputs eval_results
```

A single run:

```bash
python main.py --benchmark videomme --model qwen3_vl --eval_method remove_correct_add_none \
    --video_dir /path/to/Video-MME/data --max_frames 64 --use_logprobs --num_gpus 2
```

- `--model` values used in the paper are `qwen2_vl`, `qwen25_vl`, `qwen3_vl`, `qwen25_omni`, `qwen3_omni`, `gemma3_4b`, `gemma3_12b`, `mimo_vl`, `internvl3` and `internvl3_5`. API models use `api_models` with `--model_path gemini-2.5-flash|gemini-2.5-pro|gpt-4o --api_base <url> --api_key <key>` (any OpenAI-compatible endpoint; GPT-4o also needs `--max_image_long_side 512`).
- Inference only saves raw responses, and logprobs with `--use_logprobs`, to `outputs/<benchmark>/<model>/<eval_method>/`. Answer parsing and metrics are done by post-processing.
- With `--use_logprobs`, `--max_new_tokens` is ignored and each model's own default applies: 2048 tokens for `qwen3_vl` and InternVL, which the CoT runs used, and 128 for the others.

### Settings ↔ code

| Paper | Metric | `--eval_method` | Prompt |
|---|---|---|---|
| Baseline | ACC | `baseline` / `baseline_cot` | [eval/baseline.py](eval/baseline.py) / [eval/baseline_cot.py](eval/baseline_cot.py) |
| Multi-choice detection | MCDR | `remove_correct_add_none` / `..._cot` | [eval/remove_correct_add_none.py](eval/remove_correct_add_none.py) / [eval/remove_correct_add_none_cot.py](eval/remove_correct_add_none_cot.py) |
| Open-ended detection | OEDR | `remove_correct_force_none` / `..._cot` | [eval/remove_correct_force_none.py](eval/remove_correct_force_none.py) / [eval/remove_correct_force_none_cot.py](eval/remove_correct_force_none_cot.py) |
| Unprompted detection | UDR | `remove_correct` / `remove_correct_cot` | [eval/remove_correct.py](eval/remove_correct.py) / [eval/remove_correct_cot.py](eval/remove_correct_cot.py) |

| Step | Code |
|---|---|
| Inference (prompting, multi-GPU, resume) | [main.py](main.py), [eval/base.py](eval/base.py), [models/](models/) |
| Answer extraction with Qwen3-1.7B (CoT, API models) | [postprocess/extract_answer_llm.py](postprocess/extract_answer_llm.py) |
| ACC / MCDR / OEDR / UDR | [postprocess/compute_metrics.py](postprocess/compute_metrics.py) |
| Temporal subsets (Table 3) | [postprocess/temporal_breakdown.py](postprocess/temporal_breakdown.py) |

Notes:
- UDR counts every response without a valid option letter. Check those responses manually, because some are not abstentions: for example an out-of-range letter, or a truncated CoT response.
- A few runs are incomplete. Gemini-2.5-Flash has 2619/2700 samples on Video-MME and 486–487/500 on EgoSchema, and Gemini-2.5-Pro has 456–500/500 on EgoSchema (failed API requests). Qwen3-Omni `baseline` has 2692/2700 on Video-MME. Metrics are computed over the available samples.

## Citation

```bibtex
@inproceedings{emnlp-2026-mllm-absent-answer-detection,
    title = "When No Answer Is Correct: Diagnosing Absent Answer Detection for {MLLM}s in Video Understanding",
    author = "Wang, Yiheng  and
      Lin, Yueqian  and
      Zhu, Lichen  and
      Liu, Yudong  and
      Li, Hai Helen  and
      Chen, Yiran",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    year = "2026",
    pages = "22615--22625",
}
```
