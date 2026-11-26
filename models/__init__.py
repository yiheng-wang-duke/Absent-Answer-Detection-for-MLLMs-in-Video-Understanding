from .llava_video import LLaVAVideoModel
from .qwen2_vl import Qwen2VLModel
from .qwen25_vl import Qwen25VLModel
from .qwen3_vl import Qwen3VLModel
from .qwen25_omni import Qwen25OmniModel
from .qwen3_omni import Qwen3OmniModel
from .gemma3 import Gemma3_12BModel, Gemma3_4BModel, Gemma3_27BModel
from .litellm_model import LiteLLMModel
from .internvl import InternVL25Model, InternVL3Model, InternVL35Model
from .video_r1 import VideoR1Model
from .videochat_flash import VideoChatFlashModel
from .mimo_vl import MimoVLModel
from .keye_vl import KeyeVLModel
from .vllm_model import VLLMModel

MODELS = {
    "llava_video": LLaVAVideoModel,
    "qwen2_vl": Qwen2VLModel,
    "qwen25_vl": Qwen25VLModel,
    "qwen3_vl": Qwen3VLModel,
    "qwen25_omni": Qwen25OmniModel,
    "qwen3_omni": Qwen3OmniModel,
    "gemma3_12b": Gemma3_12BModel,
    "gemma3_4b": Gemma3_4BModel,
    "gemma3_27b": Gemma3_27BModel,
    "api_models": LiteLLMModel,
    "internvl2_5": InternVL25Model,
    "internvl3": InternVL3Model,
    "internvl3_5": InternVL35Model,
    "video_r1": VideoR1Model,
    "videochat_flash": VideoChatFlashModel,
    "mimo_vl": MimoVLModel,
    "keye_vl": KeyeVLModel,
    "vllm": VLLMModel,
}


def build_model(name: str, **kwargs):
    """Build and load a model by name.

    Args:
        name: Model key from MODELS registry.
        **kwargs: Passed to the model constructor (model_path, max_frames, device, etc.).
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODELS.keys())}")
    model = MODELS[name](**kwargs)
    model.load_model()
    return model
