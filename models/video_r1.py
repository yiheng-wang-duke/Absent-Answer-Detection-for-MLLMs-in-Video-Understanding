from .qwen25_vl import Qwen25VLModel


class VideoR1Model(Qwen25VLModel):
    """Video-R1-7B wrapper.

    Built on Qwen2.5-VL architecture; identical loading and inference.
    Note: This is a "thinking" model that may produce <think>...</think>
    reasoning before <answer>...</answer>. The raw response is returned.
    """

    name = "video_r1"

    def __init__(self, model_path: str = "Video-R1/Video-R1-7B",
                 max_frames: int = 32, device: str = "cuda:0", **kwargs):
        super().__init__(model_path, max_frames, device)
