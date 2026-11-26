import math
import logging
import torch
import subprocess
import json
from .base import BaseVideoLLM

logger = logging.getLogger(__name__)


class Qwen25OmniModel(BaseVideoLLM):
    """Qwen2.5-Omni-7B wrapper.

    Uses native video input via local path.
    Qwen2.5-Omni uses the Thinker-only variant for text generation (no speech).
    Video frames are sampled via the fps parameter in apply_chat_template.
    Each instance loads the full model on a single GPU (self.device),
    enabling multi-GPU data parallelism.
    """

    name = "qwen25_omni"
    SYSTEM_PROMPT = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )

    def __init__(self, model_path: str = "Qwen/Qwen2.5-Omni-7B",
                 max_frames: int = 32, device: str = "cuda:0", **kwargs):
        super().__init__(model_path, max_frames, device)
        self.use_audio_in_video = True

    def load_model(self):
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

        logger.info(f"Loading Qwen2_5OmniThinkerForConditionalGeneration from {self.model_path} -> {self.device}")

        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(self.device)
        self.model.eval()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)

        # Monkey-patch sample_frames to clamp indices within valid range.
        # Upstream torch.arange can produce an index == total_num_frames due to
        # floating-point drift, causing IndexError in torchcodec.
        _orig_sample_frames = self.processor.video_processor.sample_frames
        def _safe_sample_frames(metadata, **kw):
            indices = _orig_sample_frames(metadata, **kw)
            return indices.clamp(max=metadata.total_num_frames - 1)
        self.processor.video_processor.sample_frames = _safe_sample_frames

        logger.info(f"{self.name} loaded on {self.device}")

    def _build_messages(self, video_path: str, prompt: str) -> list[dict]:
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

    @staticmethod
    def _get_video_duration(video_path: str) -> float:
        """Get video duration in seconds using ffprobe."""
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", video_path,
        ]
        out = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(out.stdout)
        return float(info["format"]["duration"])

    def _prepare_inputs(self, messages: list[dict], video_path: str | None = None):
        kwargs = dict(
            load_audio_from_video=self.use_audio_in_video,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        )
        if video_path is not None:
            duration = self._get_video_duration(video_path)
            # fps that yields exactly max_frames, processor clamps via min_frames/max_frames
            kwargs["fps"] = self.max_frames / max(duration, 1e-6)
        inputs = self.processor.apply_chat_template(messages, **kwargs)
        inputs = inputs.to(self.device)
        return inputs

    def generate(self, video_path: str, prompt: str, max_new_tokens: int = 128) -> str:
        messages = self._build_messages(video_path, prompt)
        inputs = self._prepare_inputs(messages, video_path)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            gen_output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_audio_in_video=self.use_audio_in_video,
            )

        seqs = gen_output.sequences if hasattr(gen_output, "sequences") else gen_output
        output_ids = seqs[:, input_len:]
        response = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        del inputs, gen_output, output_ids
        torch.cuda.empty_cache()

        return response

    def generate_with_logprobs(
        self, video_path: str, prompt: str, option_tokens: list[str]
    ) -> dict:
        messages = self._build_messages(video_path, prompt)
        inputs = self._prepare_inputs(messages, video_path)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # (1, vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1)

        result_logprobs = {}
        for token_str in option_tokens:
            token_ids = self.processor.tokenizer.encode(
                token_str, add_special_tokens=False
            )
            if token_ids:
                result_logprobs[token_str] = log_probs[0, token_ids[0]].item()
            else:
                result_logprobs[token_str] = float("-inf")

        del inputs, outputs, logits, log_probs
        torch.cuda.empty_cache()

        response = self.generate(video_path, prompt)

        probs = {k: math.exp(v) for k, v in result_logprobs.items()}
        confidence = max(probs.values()) if probs else 0.0

        return {
            "response": response,
            "logprobs": result_logprobs,
            "probs": probs,
            "confidence": confidence,
        }

    def _build_text_only_messages(self, prompt: str) -> list[dict]:
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ]

    def _prepare_text_only_inputs(self, messages: list[dict]):
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self.device)
        return inputs

    def generate_text_only(self, prompt: str, max_new_tokens: int = 128) -> str:
        messages = self._build_text_only_messages(prompt)
        inputs = self._prepare_text_only_inputs(messages)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            gen_output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_audio_in_video=False,
            )

        seqs = gen_output.sequences if hasattr(gen_output, "sequences") else gen_output
        output_ids = seqs[:, input_len:]
        response = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        del inputs, gen_output, output_ids
        torch.cuda.empty_cache()

        return response

    def generate_text_only_with_logprobs(
        self, prompt: str, option_tokens: list[str]
    ) -> dict:
        messages = self._build_text_only_messages(prompt)
        inputs = self._prepare_text_only_inputs(messages)

        with torch.inference_mode():
            forward_keys = {'input_ids', 'attention_mask', 'position_ids'}
            filtered_inputs = {k: v for k, v in inputs.items() if k in forward_keys}

            outputs = self.model(**filtered_inputs)
            logits = outputs.logits[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)

        result_logprobs = {}
        for token_str in option_tokens:
            token_ids = self.processor.tokenizer.encode(
                token_str, add_special_tokens=False
            )
            if token_ids:
                result_logprobs[token_str] = log_probs[0, token_ids[0]].item()
            else:
                result_logprobs[token_str] = float("-inf")

        del inputs, outputs, logits, log_probs
        torch.cuda.empty_cache()

        response = self.generate_text_only(prompt)

        probs = {k: math.exp(v) for k, v in result_logprobs.items()}
        confidence = max(probs.values()) if probs else 0.0

        return {
            "response": response,
            "logprobs": result_logprobs,
            "probs": probs,
            "confidence": confidence,
        }
