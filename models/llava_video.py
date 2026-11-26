import math
import logging
import torch
from .base import BaseVideoLLM

logger = logging.getLogger(__name__)


class LLaVAVideoModel(BaseVideoLLM):
    """LLaVA-Video-7B-Qwen2 wrapper.

    Uses the llava library (LLaVA-NeXT).
    Frames are extracted via decord, processed as images.
    Conv template: qwen_1_5.
    Each instance loads the full model on a single GPU (self.device).
    """

    name = "llava_video"

    def __init__(self, model_path: str = "lmms-lab/LLaVA-Video-7B-Qwen2",
                 max_frames: int = 32, device: str = "cuda:0", **kwargs):
        super().__init__(model_path, max_frames, device)
        self.conv_mode = "qwen_1_5"

    def load_model(self):
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        model_name = get_model_name_from_path(self.model_path)
        logger.info(f"Loading LLaVA-Video from {self.model_path} -> {self.device}")

        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                model_path=self.model_path,
                model_base=None,
                model_name=model_name,
                device_map=self.device,
                torch_dtype="bfloat16",
            )
        )
        self.model.eval()
        logger.info(f"LLaVA-Video loaded on {self.device}")

    def _prepare_inputs(self, video_path: str, prompt: str):
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from utils.video_utils import extract_frames

        frames = extract_frames(video_path, max_frames=self.max_frames)

        image_tensor = process_images(frames, self.image_processor, self.model.config)
        image_tensor = image_tensor.to(self.device, dtype=torch.bfloat16)

        conv = conv_templates[self.conv_mode].copy()
        image_tokens = DEFAULT_IMAGE_TOKEN * len(frames)
        conv.append_message(conv.roles[0], image_tokens + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        return input_ids, image_tensor, len(frames)

    def generate(self, video_path: str, prompt: str, max_new_tokens: int = 128) -> str:
        input_ids, image_tensor, num_frames = self._prepare_inputs(video_path, prompt)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                modalities=["video"],
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        response = self.tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()

        del input_ids, image_tensor, output_ids
        torch.cuda.empty_cache()

        return response

    def generate_with_logprobs(
        self, video_path: str, prompt: str, option_tokens: list[str]
    ) -> dict:
        input_ids, image_tensor, num_frames = self._prepare_inputs(video_path, prompt)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                images=image_tensor,
                modalities=["video"],
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

        del input_ids, image_tensor, outputs, logits, log_probs
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

    def _prepare_text_only_inputs(self, prompt: str):
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        return input_ids

    def generate_text_only(self, prompt: str, max_new_tokens: int = 128) -> str:
        input_ids = self._prepare_text_only_inputs(prompt)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=None,
                modalities=["video"],
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        response = self.tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()

        del input_ids, output_ids
        torch.cuda.empty_cache()

        return response

    def generate_text_only_with_logprobs(
        self, prompt: str, option_tokens: list[str]
    ) -> dict:
        input_ids = self._prepare_text_only_inputs(prompt)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                images=None,
                modalities=["video"],
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

        del input_ids, outputs, logits, log_probs
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
