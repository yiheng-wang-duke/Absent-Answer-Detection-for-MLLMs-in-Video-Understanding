import math
import logging
import torch
from .base import BaseVideoLLM

logger = logging.getLogger(__name__)


class VideoChatFlashModel(BaseVideoLLM):
    """VideoChat-Flash-Qwen2-7B wrapper (OpenGVLab).

    Uses trust_remote_code with model.chat() API.
    Requires: transformers==4.40.1, decord, av, flash-attn.
    """

    name = "videochat_flash"

    def __init__(self, model_path: str = "OpenGVLab/VideoChat-Flash-Qwen2-7B_res448",
                 max_frames: int = 32, device: str = "cuda:0", **kwargs):
        super().__init__(model_path, max_frames, device)

    def load_model(self):
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading VideoChat-Flash from {self.model_path} -> {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True
        ).to(torch.bfloat16).to(self.device)
        self.model.eval()
        logger.info(f"{self.name} loaded on {self.device}")

    def generate(self, video_path: str, prompt: str, max_new_tokens: int = 128) -> str:
        generation_config = dict(
            do_sample=False, temperature=0.0, max_new_tokens=max_new_tokens,
            top_p=0.1, num_beams=1,
        )
        result = self.model.chat(
            video_path=video_path,
            tokenizer=self.tokenizer,
            user_prompt=prompt,
            return_history=False,
            max_num_frames=self.max_frames,
            generation_config=generation_config,
        )
        if isinstance(result, tuple):
            result = result[0]

        torch.cuda.empty_cache()
        return result.strip()

    def generate_with_logprobs(
        self, video_path: str, prompt: str, option_tokens: list[str]
    ) -> dict:
        response = self.generate(video_path, prompt)

        # VideoChat-Flash uses model.chat() which doesn't expose logits directly.
        # Attempt a manual forward pass for logprobs.
        try:
            input_ids, attention_mask, pixel_values = self._prepare_manual_inputs(
                video_path, prompt
            )
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                )
                logits = outputs.logits[:, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1)

            result_logprobs = {}
            for token_str in option_tokens:
                token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
                if token_ids:
                    result_logprobs[token_str] = log_probs[0, token_ids[0]].item()
                else:
                    result_logprobs[token_str] = float("-inf")

            del input_ids, attention_mask, pixel_values, outputs, logits, log_probs
            torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"Could not get logprobs for VideoChat-Flash: {e}. "
                           "Returning uniform probabilities.")
            result_logprobs = {t: 0.0 for t in option_tokens}

        probs = {k: math.exp(v) for k, v in result_logprobs.items()}
        confidence = max(probs.values()) if probs else 0.0

        return {
            "response": response,
            "logprobs": result_logprobs,
            "probs": probs,
            "confidence": confidence,
        }

    def _prepare_manual_inputs(self, video_path: str, prompt: str):
        from utils.video_utils import extract_frames

        frames = extract_frames(video_path, max_frames=self.max_frames)
        image_processor = self.model.get_vision_tower().image_processor

        pixel_values = image_processor(images=frames, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device, dtype=torch.bfloat16)

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids)

        return input_ids, attention_mask, pixel_values

    def generate_text_only(self, prompt: str, max_new_tokens: int = 128) -> str:
        generation_config = dict(
            do_sample=False, temperature=0.0, max_new_tokens=max_new_tokens,
            top_p=0.1, num_beams=1,
        )
        try:
            result = self.model.chat(
                video_path=None,
                tokenizer=self.tokenizer,
                user_prompt=prompt,
                return_history=False,
                max_num_frames=0,
                generation_config=generation_config,
            )
        except TypeError:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                )
            result = self.tokenizer.decode(
                output_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )

        if isinstance(result, tuple):
            result = result[0]

        torch.cuda.empty_cache()
        return result.strip()

    def generate_text_only_with_logprobs(
        self, prompt: str, option_tokens: list[str]
    ) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )
            logits = outputs.logits[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)

        result_logprobs = {}
        for token_str in option_tokens:
            token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
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
