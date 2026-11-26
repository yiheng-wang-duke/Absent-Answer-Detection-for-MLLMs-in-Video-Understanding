from .qwen25_vl import Qwen25VLModel


class MimoVLModel(Qwen25VLModel):
    """MiMo-VL-7B-RL wrapper (Xiaomi).

    Uses Qwen2_5_VLForConditionalGeneration architecture; identical API.
    """

    name = "mimo_vl"

    def __init__(self, model_path: str = "XiaomiMiMo/MiMo-VL-7B-RL",
                 max_frames: int = 32, device: str = "cuda:0", **kwargs):
        super().__init__(model_path, max_frames, device)
