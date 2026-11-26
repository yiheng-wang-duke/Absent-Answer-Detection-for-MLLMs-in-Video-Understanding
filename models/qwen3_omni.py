import math
import logging
import torch
from .base import BaseVideoLLM

logger = logging.getLogger(__name__)


class Qwen3OmniModel(BaseVideoLLM):
    """Qwen3-Omni-30B-A3B-Instruct wrapper.

    Uses native video input with audio extracted from the video track.
    Video frames are sampled via nframes parameter in the message dict,
    and audio is automatically extracted when use_audio_in_video=True.
    Media processing is handled by qwen_omni_utils.process_mm_info.

    The 30B MoE model (~60GB in BF16) is loaded with device_map="auto"
    for automatic multi-GPU model parallelism.  Run with --num_gpus 1
    so the evaluator uses a single worker while the model spans all
    visible GPUs.
    """

    name = "qwen3_omni"

    def __init__(self, model_path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                 max_frames: int = 32, device: str = "cuda:0", **kwargs):
        super().__init__(model_path, max_frames, device)
        self.use_audio_in_video = False  # Temporarily disable audio to debug device_map issue

    def load_model(self):
        # from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        from transformers import Qwen3OmniMoeThinkerForConditionalGeneration, Qwen3OmniMoeProcessor
        logger.info(f"Loading Qwen3OmniMoeForConditionalGeneration from {self.model_path} (device_map=auto)")

        self.model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        # Disable the talker (speech output) since we only need text
        # self.model.disable_talker()
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)
        logger.info(f"{self.name} loaded with device_map=auto (input device: {self.model.device})")

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
        # Use official API: apply_chat_template handles video/audio loading
        inputs = self.processor.apply_chat_template(
            messages,
            load_audio_from_video=self.use_audio_in_video,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            fps=self.max_frames / 60.0,  # Convert max_frames to fps (assuming 60s video)
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        ).to(self.model.device)

        return inputs

    def generate(self, video_path: str, prompt: str, max_new_tokens: int = 128) -> str:
        messages = self._build_messages(video_path, prompt)
        inputs = self._prepare_inputs(messages)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            gen_output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_audio_in_video=self.use_audio_in_video,
            )

        # Handle both GenerateOutput (with .sequences) and raw tensor
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
        # Use official API for text-only inputs
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

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

        # Debug: print all input keys
        logger.info(f"Text-only input keys: {list(inputs.keys())}")

        with torch.inference_mode():
            # Filter inputs to only include keys that forward accepts
            forward_keys = {'input_ids', 'attention_mask', 'position_ids'}
            filtered_inputs = {k: v for k, v in inputs.items() if k in forward_keys}
            logger.info(f"Filtered text-only input keys: {list(filtered_inputs.keys())}")

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
