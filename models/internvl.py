import math
import logging
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from .base import BaseVideoLLM

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def _load_video(video_path, num_segments=32, input_size=448, max_num=1):
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    seg_size = float(max_frame) / num_segments
    frame_indices = np.array([
        int(seg_size / 2 + np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    frame_indices = np.clip(frame_indices, 0, max_frame)

    transform = _build_transform(input_size)
    pixel_values_list, num_patches_list = [], []

    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        tiles = _dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = torch.stack([transform(tile) for tile in tiles])
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


class InternVLModel(BaseVideoLLM):
    """InternVL base wrapper using trust_remote_code + model.chat() API.

    Supports InternVL2.5, InternVL3, and InternVL3.5.
    Requires: decord, torchvision, flash-attn (optional).
    """

    name = "internvl"

    def __init__(self, model_path: str = "OpenGVLab/InternVL2_5-8B",
                 max_frames: int = 32, device: str = "cuda:0", **kwargs):
        super().__init__(model_path, max_frames, device)

    def load_model(self):
        from transformers import AutoTokenizer, AutoModel, GenerationMixin

        logger.info(f"Loading InternVL from {self.model_path} -> {self.device}")
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).eval().to(self.device)

        # In transformers>=4.47, GenerationMixin was decoupled from
        # PreTrainedModel. The InternVL remote code expects the language
        # model (InternLM2ForCausalLM) to inherit .generate() from
        # PreTrainedModel, which is no longer the case. Patch it here.
        lm = self.model.language_model
        if not hasattr(lm, 'generate'):
            lm.__class__.__bases__ = (
                *lm.__class__.__bases__, GenerationMixin,
            )
            logger.info("Patched language_model with GenerationMixin")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False
        )
        logger.info(f"{self.name} loaded on {self.device}")

    def _load_video_inputs(self, video_path: str):
        pixel_values, num_patches_list = _load_video(
            video_path, num_segments=self.max_frames
        )
        pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
        return pixel_values, num_patches_list

    def _build_question(self, num_patches_list, prompt):
        video_prefix = ''.join(
            [f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))]
        )
        return video_prefix + prompt

    def generate(self, video_path: str, prompt: str, max_new_tokens: int = 2048) -> str:
        pixel_values, num_patches_list = self._load_video_inputs(video_path)
        question = self._build_question(num_patches_list, prompt)

        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
        response = self.model.chat(
            self.tokenizer, pixel_values, question, generation_config,
            num_patches_list=num_patches_list, history=None, return_history=False,
        )

        del pixel_values
        torch.cuda.empty_cache()
        return response.strip() if isinstance(response, str) else response[0].strip()

    def generate_with_logprobs(
        self, video_path: str, prompt: str, option_tokens: list[str]
    ) -> dict:
        pixel_values, num_patches_list = self._load_video_inputs(video_path)
        question = self._build_question(num_patches_list, prompt)

        video_prefix = ''.join(
            [f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))]
        )
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        image_tokens = IMG_CONTEXT_TOKEN * self.model.num_image_token
        num_patches = num_patches_list

        frame_prompts = []
        for i, n in enumerate(num_patches):
            frame_prompts.append(
                f'Frame{i+1}: <img>' + image_tokens * n + '</img>\n'
            )
        full_question = ''.join(frame_prompts) + prompt

        model_inputs = self.tokenizer(
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": full_question}],
                tokenize=False, add_generation_prompt=True
            ),
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.language_model(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
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

        del model_inputs, outputs, logits, log_probs
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

    def generate_text_only(self, prompt: str, max_new_tokens: int = 128) -> str:
        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
        response = self.model.chat(
            self.tokenizer, None, prompt, generation_config,
            history=None, return_history=False,
        )
        return response.strip() if isinstance(response, str) else response[0].strip()

    def generate_text_only_with_logprobs(
        self, prompt: str, option_tokens: list[str]
    ) -> dict:
        model_inputs = self.tokenizer(
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True
            ),
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.language_model(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
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

        del model_inputs, outputs, logits, log_probs
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


class InternVL25Model(InternVLModel):
    """InternVL2.5-8B wrapper."""
    name = "internvl2_5"

    def __init__(self, model_path: str = "OpenGVLab/InternVL2_5-8B",
                 max_frames: int = 32, device: str = "cuda:0", **kwargs):
        super().__init__(model_path, max_frames, device)


class InternVL3Model(InternVLModel):
    """InternVL3-8B wrapper."""
    name = "internvl3"

    def __init__(self, model_path: str = "OpenGVLab/InternVL3-8B",
                 max_frames: int = 32, device: str = "cuda:0", **kwargs):
        super().__init__(model_path, max_frames, device)


class InternVL35Model(InternVLModel):
    """InternVL3.5-8B wrapper."""
    name = "internvl3_5"

    def __init__(self, model_path: str = "OpenGVLab/InternVL3_5-8B",
                 max_frames: int = 32, device: str = "cuda:0", **kwargs):
        super().__init__(model_path, max_frames, device)
