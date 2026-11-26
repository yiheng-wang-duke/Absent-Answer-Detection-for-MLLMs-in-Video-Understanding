import math
import logging
import torch
import torch._dynamo
from .base import BaseVideoLLM

logger = logging.getLogger(__name__)

# Gemma3's _update_causal_mask accesses self.config.text_config (a
# Gemma3TextConfig), which TorchDynamo cannot trace.  Fall back to eager
# execution instead of raising on every sample.
torch._dynamo.config.suppress_errors = True


class Gemma3Model(BaseVideoLLM):
    """Gemma-3 (4B/12B/27B) multimodal wrapper for video understanding.

    Gemma 3 does not have native video input. Instead, we extract frames
    from the video and pass them as multiple images in the message content.
    Uses Gemma3ForConditionalGeneration + AutoProcessor.
    Each instance loads the full model on a single GPU (self.device).
    """

    name = "gemma3"

    def __init__(self, model_path: str = "google/gemma-3-12b-it",
                 max_frames: int = 32, device: str = "cuda:0", **kwargs):
        super().__init__(model_path, max_frames, device)

    def load_model(self):
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        logger.info(f"Loading Gemma3ForConditionalGeneration from {self.model_path} -> {self.device}")

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        logger.info(f"{self.name} loaded on {self.device}")

    def _build_messages(self, frames, prompt: str) -> list[dict]:
        """Build chat messages with extracted frames as multiple images."""
        content = []
        for frame in frames:
            content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content}]

    def _prepare_inputs(self, video_path: str, prompt: str):
        from utils.video_utils import extract_frames

        frames = extract_frames(video_path, max_frames=self.max_frames)
        messages = self._build_messages(frames, prompt)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            pad_to_multiple_of=8,
            do_pan_and_scan=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()
                  if isinstance(v, torch.Tensor)}
        return inputs

    def generate(self, video_path: str, prompt: str, max_new_tokens: int = 128) -> str:
        inputs = self._prepare_inputs(video_path, prompt)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        output_ids = generated_ids[:, input_len:]
        response = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()

        del inputs, generated_ids, output_ids
        torch.cuda.empty_cache()

        return response

    def generate_with_logprobs(
        self, video_path: str, prompt: str, option_tokens: list[str]
    ) -> dict:
        inputs = self._prepare_inputs(video_path, prompt)

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
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]

    def _prepare_text_only_inputs(self, prompt: str):
        messages = self._build_text_only_messages(prompt)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()
                  if isinstance(v, torch.Tensor)}
        return inputs

    def generate_text_only(self, prompt: str, max_new_tokens: int = 128) -> str:
        inputs = self._prepare_text_only_inputs(prompt)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        output_ids = generated_ids[:, input_len:]
        response = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()

        del inputs, generated_ids, output_ids
        torch.cuda.empty_cache()

        return response

    def generate_text_only_with_logprobs(
        self, prompt: str, option_tokens: list[str]
    ) -> dict:
        inputs = self._prepare_text_only_inputs(prompt)

        with torch.inference_mode():
            outputs = self.model(**inputs)
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


class Gemma3_4BModel(Gemma3Model):
    """Gemma-3-4B-IT wrapper."""

    name = "gemma3_4b"

    def __init__(self, model_path: str = "google/gemma-3-4b-it",
                 max_frames: int = 32, device: str = "cuda:0", **kwargs):
        super().__init__(model_path, max_frames, device)


class Gemma3_12BModel(Gemma3Model):
    """Gemma-3-12B-IT wrapper."""

    name = "gemma3_12b"

    def __init__(self, model_path: str = "google/gemma-3-12b-it",
                 max_frames: int = 32, device: str = "cuda:0", **kwargs):
        super().__init__(model_path, max_frames, device)


class Gemma3_27BModel(Gemma3Model):
    """Gemma-3-27B-IT wrapper."""

    name = "gemma3_27b"

    def __init__(self, model_path: str = "google/gemma-3-27b-it",
                 max_frames: int = 32, device: str = "cuda:0", **kwargs):
        super().__init__(model_path, max_frames, device)
