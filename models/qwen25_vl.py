from .qwen2_vl import Qwen2VLModel


class Qwen25VLModel(Qwen2VLModel):
    """Qwen2.5-VL-7B-Instruct wrapper.

    Identical API to Qwen2-VL; only the model class name differs.
    """

    name = "qwen25_vl"
    model_cls_name = "Qwen2_5_VLForConditionalGeneration"

    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 max_frames: int = 32, device: str = "cuda:0", **kwargs):
        super().__init__(model_path, max_frames, device)
