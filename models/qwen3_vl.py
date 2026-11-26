import math
import logging
import torch
from .base import BaseVideoLLM

logger = logging.getLogger(__name__)


class Qwen3VLModel(BaseVideoLLM):
    """Qwen3-VL-8B-Instruct wrapper.

    Uses native video input via local path.
    Qwen3-VL uses the new transformers processor API where apply_chat_template
    handles video loading and frame sampling directly (no qwen_vl_utils needed).
    Frame count is controlled via num_frames kwarg to apply_chat_template.
    Each instance loads the full model on a single GPU (self.device).
    """

    name = "qwen3_vl"
    model_cls_name = "Qwen3VLForConditionalGeneration"

    def __init__(self, model_path: str = "Qwen/Qwen3-VL-8B-Instruct",
                 max_frames: int = 32, device: str = "cuda:0", **kwargs):
        super().__init__(model_path, max_frames, device)

    def load_model(self):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        logger.info(f"Loading Qwen3VLForConditionalGeneration from {self.model_path} -> {self.device}")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        # Clear default fps so it doesn't conflict with num_frames in apply_chat_template
        self.processor.video_processor.fps = None
        logger.info(f"{self.name} loaded on {self.device}")

    def _build_messages(self, video_path: str, prompt: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _prepare_inputs(self, messages: list[dict]):
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            num_frames=self.max_frames,
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self.device)
        return inputs

    def generate(self, video_path: str, prompt: str, max_new_tokens: int = 2048) -> str:
        messages = self._build_messages(video_path, prompt)
        inputs = self._prepare_inputs(messages)
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
        messages = self._build_messages(video_path, prompt)
        inputs = self._prepare_inputs(messages)

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

    def _prepare_text_only_inputs(self, messages: list[dict]):
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self.device)
        return inputs

    def generate_text_only(self, prompt: str, max_new_tokens: int = 128) -> str:
        messages = self._build_text_only_messages(prompt)
        inputs = self._prepare_text_only_inputs(messages)
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
        messages = self._build_text_only_messages(prompt)
        inputs = self._prepare_text_only_inputs(messages)

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
