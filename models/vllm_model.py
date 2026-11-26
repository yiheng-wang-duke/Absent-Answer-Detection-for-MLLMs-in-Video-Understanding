import math
import logging
import os
import numpy as np
from .base import BaseVideoLLM

logger = logging.getLogger(__name__)


class VLLMModel(BaseVideoLLM):
    """Generic vLLM wrapper for any vLLM-supported vision-language model.

    Uses vLLM's offline LLM engine for accelerated inference with
    PagedAttention, continuous batching, and tensor parallelism.

    Supports both single-sample (BaseVideoLLM interface) and batch inference.
    """

    name = "vllm"

    def __init__(self, model_path: str, max_frames: int = 32,
                 device: str = "cuda:0",
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9,
                 max_model_len: int = None,
                 batch_size: int = 32,
                 **kwargs):
        super().__init__(model_path, max_frames, device)
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.batch_size = batch_size

    def load_model(self):
        from vllm import LLM

        logger.info(
            f"Loading vLLM model from {self.model_path} "
            f"(tp={self.tensor_parallel_size}, "
            f"mem={self.gpu_memory_utilization})"
        )

        kwargs = {
            "model": self.model_path,
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            # Use video format: 1 video per prompt instead of N images
            "limit_mm_per_prompt": {"video": 1},
        }
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len

        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        logger.info(f"vLLM model loaded: {self.model_path}")

    # ------------------------------------------------------------------
    # Video frame helpers
    # ------------------------------------------------------------------
    def _extract_video(self, video_path: str):
        """Extract frames and metadata for vLLM video input.

        Returns (frames_np, metadata) where frames_np is (N, H, W, 3)
        and metadata contains fps, duration, frames_indices, etc.
        """
        from utils.video_utils import extract_frames_with_metadata
        return extract_frames_with_metadata(
            video_path, max_frames=self.max_frames
        )

    def _build_video_messages(self, video_path: str, prompt: str):
        """Build chat messages with video as a single video input.

        Returns chat messages and (frames_np, metadata) tuple.
        vLLM requires video metadata (fps, duration, frames_indices)
        for Qwen2/3-VL models.
        """
        frames_np, metadata = self._extract_video(video_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{video_path}",
                        "nframes": len(frames_np),
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return messages, frames_np, metadata

    def _build_text_messages(self, prompt: str):
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]

    def _apply_chat_template(self, messages):
        """Apply the model's chat template to format the prompt."""
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _build_video_input(self, video_path: str, prompt: str):
        """Build a complete vLLM input dict for a video prompt."""
        messages, frames_np, metadata = self._build_video_messages(
            video_path, prompt
        )
        formatted = self._apply_chat_template(messages)
        # vLLM expects video as list of (np.ndarray, metadata_dict) tuples
        return {
            "prompt": formatted,
            "multi_modal_data": {"video": [(frames_np, metadata)]},
        }

    def _build_text_input(self, prompt: str):
        """Build a complete vLLM input dict for a text-only prompt."""
        messages = self._build_text_messages(prompt)
        formatted = self._apply_chat_template(messages)
        return {"prompt": formatted}

    # ------------------------------------------------------------------
    # Single-sample interface (BaseVideoLLM compatibility)
    # ------------------------------------------------------------------
    def generate(self, video_path: str, prompt: str,
                 max_new_tokens: int = 128) -> str:
        from vllm import SamplingParams

        inp = self._build_video_input(video_path, prompt)
        params = SamplingParams(max_tokens=max_new_tokens, temperature=0)
        outputs = self.llm.generate([inp], sampling_params=params)
        return outputs[0].outputs[0].text.strip()

    def generate_with_logprobs(
        self, video_path: str, prompt: str, option_tokens: list[str]
    ) -> dict:
        from vllm import SamplingParams

        inp = self._build_video_input(video_path, prompt)

        # First: get logprobs for option tokens (generate 1 token with logprobs)
        logprob_params = SamplingParams(
            max_tokens=1, temperature=0, logprobs=20,
        )
        logprob_outputs = self.llm.generate(
            [inp], sampling_params=logprob_params,
        )
        result_logprobs = self._extract_option_logprobs(
            logprob_outputs[0], option_tokens
        )

        # Second: get full response
        gen_params = SamplingParams(
            max_tokens=128, temperature=0,
        )
        gen_outputs = self.llm.generate(
            [inp], sampling_params=gen_params,
        )
        response = gen_outputs[0].outputs[0].text.strip()

        probs = {k: math.exp(v) for k, v in result_logprobs.items()}
        confidence = max(probs.values()) if probs else 0.0

        return {
            "response": response,
            "logprobs": result_logprobs,
            "probs": probs,
            "confidence": confidence,
        }

    def generate_text_only(self, prompt: str,
                           max_new_tokens: int = 128) -> str:
        from vllm import SamplingParams

        inp = self._build_text_input(prompt)
        params = SamplingParams(max_tokens=max_new_tokens, temperature=0)
        outputs = self.llm.generate([inp], sampling_params=params)
        return outputs[0].outputs[0].text.strip()

    def generate_text_only_with_logprobs(
        self, prompt: str, option_tokens: list[str]
    ) -> dict:
        from vllm import SamplingParams

        inp = self._build_text_input(prompt)

        logprob_params = SamplingParams(
            max_tokens=1, temperature=0, logprobs=20,
        )
        logprob_outputs = self.llm.generate(
            [inp], sampling_params=logprob_params,
        )
        result_logprobs = self._extract_option_logprobs(
            logprob_outputs[0], option_tokens
        )

        response = self.generate_text_only(prompt)

        probs = {k: math.exp(v) for k, v in result_logprobs.items()}
        confidence = max(probs.values()) if probs else 0.0

        return {
            "response": response,
            "logprobs": result_logprobs,
            "probs": probs,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------
    # Batch inference (vLLM-specific, main speedup source)
    # ------------------------------------------------------------------
    def generate_batch(self, items: list[dict],
                       max_new_tokens: int = 128) -> list[str]:
        """Batch generation for multiple samples."""
        from vllm import SamplingParams

        prompts = []
        for item in items:
            video_path = item.get("video_path")
            if video_path and os.path.exists(video_path):
                prompts.append(
                    self._build_video_input(video_path, item["prompt"])
                )
            else:
                prompts.append(self._build_text_input(item["prompt"]))

        params = SamplingParams(max_tokens=max_new_tokens, temperature=0)
        outputs = self.llm.generate(prompts, sampling_params=params)
        return [o.outputs[0].text.strip() for o in outputs]

    def generate_batch_with_logprobs(
        self, items: list[dict], max_new_tokens: int = 128
    ) -> list[dict]:
        """Batch generation with logprobs for option tokens."""
        from vllm import SamplingParams

        # Step 1: build all prompts
        prompt_inputs = []
        for item in items:
            video_path = item.get("video_path")
            if video_path and os.path.exists(video_path):
                prompt_inputs.append(
                    self._build_video_input(video_path, item["prompt"])
                )
            else:
                prompt_inputs.append(self._build_text_input(item["prompt"]))

        # Step 2: get logprobs (1 token)
        logprob_params = SamplingParams(
            max_tokens=1, temperature=0, logprobs=20,
        )
        logprob_outputs = self.llm.generate(
            prompt_inputs, sampling_params=logprob_params
        )

        # Step 3: get full responses
        gen_params = SamplingParams(
            max_tokens=max_new_tokens, temperature=0,
        )
        gen_outputs = self.llm.generate(
            prompt_inputs, sampling_params=gen_params
        )

        # Step 4: combine results
        results = []
        for i, item in enumerate(items):
            option_tokens = item.get("option_tokens", [])
            result_logprobs = self._extract_option_logprobs(
                logprob_outputs[i], option_tokens
            )
            response = gen_outputs[i].outputs[0].text.strip()

            probs = {k: math.exp(v) for k, v in result_logprobs.items()}
            confidence = max(probs.values()) if probs else 0.0

            results.append({
                "response": response,
                "logprobs": result_logprobs,
                "probs": probs,
                "confidence": confidence,
            })

        return results

    # ------------------------------------------------------------------
    # Logprob extraction helper
    # ------------------------------------------------------------------
    def _extract_option_logprobs(self, output, option_tokens: list[str]) -> dict:
        """Extract log-probabilities for option tokens from vLLM output."""
        result = {}
        first_token_logprobs = {}

        if output.outputs[0].logprobs:
            lp_dict = output.outputs[0].logprobs[0]
            for token_id, logprob_obj in lp_dict.items():
                decoded = logprob_obj.decoded_token
                if decoded is not None:
                    first_token_logprobs[decoded.strip()] = logprob_obj.logprob

        for token_str in option_tokens:
            if token_str in first_token_logprobs:
                result[token_str] = first_token_logprobs[token_str]
            else:
                token_ids = self.tokenizer.encode(
                    token_str, add_special_tokens=False
                )
                found = False
                if token_ids and output.outputs[0].logprobs:
                    tid = token_ids[0]
                    lp_dict = output.outputs[0].logprobs[0]
                    if tid in lp_dict:
                        result[token_str] = lp_dict[tid].logprob
                        found = True
                if not found:
                    result[token_str] = float("-inf")

        return result
