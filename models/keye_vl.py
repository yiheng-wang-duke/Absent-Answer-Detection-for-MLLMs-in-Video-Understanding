import math
import logging
import torch
from .base import BaseVideoLLM

logger = logging.getLogger(__name__)


class KeyeVLModel(BaseVideoLLM):
    """Keye-VL-1.5-8B wrapper (Kwai/Tencent).

    Uses trust_remote_code + keye_vl_utils for vision processing.
    Requires: keye-vl-utils==1.5.2, flash-attn.
    """

    name = "keye_vl"

    def __init__(self, model_path: str = "Kwai-Keye/Keye-VL-1_5-8B",
                 max_frames: int = 32, device: str = "cuda:0", **kwargs):
        super().__init__(model_path, max_frames, device)

    def load_model(self):
        from transformers import AutoModel, AutoProcessor

        logger.info(f"Loading Keye-VL from {self.model_path} -> {self.device}")
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        logger.info(f"{self.name} loaded on {self.device}")

    def _build_messages(self, video_path: str, prompt: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{video_path}",
                        "max_frames": self.max_frames,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _prepare_inputs(self, messages: list[dict]):
        from keye_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, mm_kwargs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **mm_kwargs,
        )
        inputs = inputs.to(self.device)
        return inputs

    def generate(self, video_path: str, prompt: str, max_new_tokens: int = 128) -> str:
        messages = self._build_messages(video_path, prompt)
        inputs = self._prepare_inputs(messages)
        input_len = inputs.input_ids.shape[1]

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
            logits = outputs.logits[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)

        result_logprobs = {}
        tokenizer = self.processor.tokenizer
        for token_str in option_tokens:
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
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
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], padding=True, return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        return inputs

    def generate_text_only(self, prompt: str, max_new_tokens: int = 128) -> str:
        messages = self._build_text_only_messages(prompt)
        inputs = self._prepare_text_only_inputs(messages)
        input_len = inputs.input_ids.shape[1]

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
        tokenizer = self.processor.tokenizer
        for token_str in option_tokens:
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
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
