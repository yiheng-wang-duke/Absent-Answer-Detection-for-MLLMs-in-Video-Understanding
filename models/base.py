import math
from abc import ABC, abstractmethod


class BaseVideoLLM(ABC):
    """Abstract base class for video LLM wrappers.

    All models expose two interfaces:
    - generate(): video + prompt -> response text
    - generate_with_logprobs(): video + prompt -> response + per-option log-probabilities

    Each model instance is loaded on a single GPU specified by `device`.
    For data parallelism, create one instance per GPU.
    """

    name: str = ""

    def __init__(self, model_path: str, max_frames: int = 32,
                 device: str = "cuda:0", **kwargs):
        self.model_path = model_path
        self.max_frames = max_frames
        self.device = device

    @abstractmethod
    def load_model(self):
        """Load the model and processor onto self.device. Called once."""
        ...

    @abstractmethod
    def generate(self, video_path: str, prompt: str, max_new_tokens: int = 128) -> str:
        """Generate a text response given a video and prompt."""
        ...

    @abstractmethod
    def generate_with_logprobs(
        self, video_path: str, prompt: str, option_tokens: list[str]
    ) -> dict:
        """Generate response and return log-probabilities for option tokens.

        Returns:
            Dict with: response, logprobs (letter->logp), probs (letter->prob),
            confidence (max prob).
        """
        ...

    @abstractmethod
    def generate_text_only(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Generate a text response given only a text prompt (no visual input)."""
        ...

    @abstractmethod
    def generate_text_only_with_logprobs(
        self, prompt: str, option_tokens: list[str]
    ) -> dict:
        """Generate response and return log-probabilities without visual input.

        Returns:
            Dict with: response, logprobs (letter->logp), probs (letter->prob),
            confidence (max prob).
        """
        ...
