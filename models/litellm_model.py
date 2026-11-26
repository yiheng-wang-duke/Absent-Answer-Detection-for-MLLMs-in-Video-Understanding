import base64
import io
import math
import logging

import requests

from .base import BaseVideoLLM

logger = logging.getLogger(__name__)


class LiteLLMModel(BaseVideoLLM):
    """API-based video LLM wrapper using raw requests to OpenAI-compatible API.

    Sends frames as base64-encoded JPEG images via the chat/completions
    endpoint (works with LiteLLM proxy, OpenAI, Gemini, etc.).

    No local GPU is needed — inference happens on the remote server.
    """

    name = "api_models"

    def __init__(self, model_path: str = "gemini-2.5-flash",
                 max_frames: int = 32, device: str = "cuda:0",
                 api_base: str = None, api_key: str = None,
                 max_image_long_side: int = 0,
                 reasoning_effort: str = None, **kwargs):
        super().__init__(model_path, max_frames, device)
        self.api_base = api_base.rstrip("/") if api_base else ""
        self.api_key = api_key or ""
        # Resize frames so the longest edge is <= this many pixels (0 = no resize).
        # Required for gpt-4o (long side <= 512).
        self.max_image_long_side = int(max_image_long_side or 0)
        # reasoning_effort valid for this model. None -> auto-resolve from name.
        self.reasoning_effort = self._resolve_reasoning_effort(
            model_path, reasoning_effort
        )

    @staticmethod
    def _resolve_reasoning_effort(model_path: str, override):
        """Pick a reasoning_effort value valid for the target model.

        Backends behind the LiteLLM proxy disagree on this param:
        - OpenAI GPT chat models (gpt-4o, gpt-4o-mini) reject reasoning_effort
          entirely -> must omit it (None).
        - gemini-2.5-pro is thinking-only and rejects "none" (budget 0) -> "low".
        - gemini-2.5-flash supports disabling thinking via "none".

        An explicit (non-None) override from the caller is honored as-is.
        """
        if override is not None:
            return override
        m = (model_path or "").lower()
        if m.startswith("gpt-") or m.startswith("chatgpt") or m.startswith("o1") \
                or m.startswith("o3") or m.startswith("o4"):
            # OpenAI models: gpt-4o family reject the param; reasoning models
            # (o-series) accept it but we leave them at server default here.
            return None
        if "gemini-2.5-pro" in m or "gemini-3" in m:
            # Thinking-only Gemini models cannot disable thinking ("none").
            return "low"
        # gemini-2.5-flash and similar: disable thinking for cheap MCQ answers.
        return "none"

    def load_model(self):
        """Set up a persistent requests session."""
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        # Per-call token usage (updated after each _call_api)
        self.last_usage = {}
        # Cumulative token usage across all calls
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        logger.info(
            f"API model ready: model={self.model_path}, "
            f"api_base={self.api_base}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _encode_frames(self, video_path: str) -> list[str]:
        """Extract frames from video and return base64 JPEG strings."""
        from utils.video_utils import extract_frames

        frames = extract_frames(video_path, max_frames=self.max_frames)
        b64_list = []
        for frame in frames:
            frame = self._resize_frame(frame)
            buf = io.BytesIO()
            frame.save(buf, format="JPEG", quality=85)
            b64_list.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        return b64_list

    def _resize_frame(self, frame):
        """Downscale a PIL frame so its longest edge <= max_image_long_side.

        Preserves aspect ratio. No-op when max_image_long_side <= 0 or the
        frame already fits.
        """
        limit = self.max_image_long_side
        if limit <= 0:
            return frame
        w, h = frame.size
        long_side = max(w, h)
        if long_side <= limit:
            return frame
        scale = limit / float(long_side)
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        from PIL import Image
        return frame.resize(new_size, Image.BICUBIC)

    def _build_content(self, b64_frames: list[str], prompt: str) -> list[dict]:
        """Build multimodal content list (images + text)."""
        content: list[dict] = [{"type": "text", "text": prompt}]
        for b64 in b64_frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        return content

    def _call_api(self, messages: list[dict], max_tokens: int = 4096) -> dict:
        """Send a chat completion request and return the parsed JSON."""
        payload = {
            "model": self.model_path,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        # Only send reasoning_effort when the target model accepts it.
        # gpt-4o/gpt-4o-mini reject the param; gemini-2.5-pro rejects "none".
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort

        resp = self.session.post(
            f"{self.api_base}/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        # Track token usage
        usage = data.get("usage", {})
        self.last_usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        for k in self.total_usage:
            self.total_usage[k] += self.last_usage.get(k, 0)

        return data

    # ------------------------------------------------------------------
    # Video + text generation
    # ------------------------------------------------------------------
    def generate(self, video_path: str, prompt: str,
                 max_new_tokens: int = 128) -> str:
        b64_frames = self._encode_frames(video_path)
        content = self._build_content(b64_frames, prompt)
        messages = [{"role": "user", "content": content}]

        data = self._call_api(messages, max_tokens=max_new_tokens)
        choice = data["choices"][0]
        text = choice["message"].get("content")
        if text is None:
            logger.warning(f"content is None. finish_reason={choice.get('finish_reason')}, message={choice['message']}")
        return text.strip()

    def generate_with_logprobs(
        self, video_path: str, prompt: str, option_tokens: list[str]
    ) -> dict:
        # API models typically don't support per-token logprobs;
        # generate normally and return uniform confidence as fallback.
        response = self.generate(video_path, prompt)
        return self._make_logprob_result(response, option_tokens)

    # ------------------------------------------------------------------
    # Text-only generation
    # ------------------------------------------------------------------
    def generate_text_only(self, prompt: str,
                           max_new_tokens: int = 128) -> str:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        data = self._call_api(messages, max_tokens=max_new_tokens)
        return data["choices"][0]["message"].get("content", "").strip()

    def generate_text_only_with_logprobs(
        self, prompt: str, option_tokens: list[str]
    ) -> dict:
        response = self.generate_text_only(prompt)
        return self._make_logprob_result(response, option_tokens)

    # ------------------------------------------------------------------
    # Logprob fallback
    # ------------------------------------------------------------------
    @staticmethod
    def _make_logprob_result(response: str, option_tokens: list[str]) -> dict:
        """Build a logprob result dict with uniform fallback probabilities."""
        uniform_lp = math.log(1.0 / max(len(option_tokens), 1))
        logprobs = {t: uniform_lp for t in option_tokens}
        probs = {t: math.exp(uniform_lp) for t in option_tokens}
        return {
            "response": response,
            "logprobs": logprobs,
            "probs": probs,
            "confidence": max(probs.values()) if probs else 0.0,
        }
