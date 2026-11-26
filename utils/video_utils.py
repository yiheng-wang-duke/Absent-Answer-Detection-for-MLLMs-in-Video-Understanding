import os
import numpy as np
from PIL import Image

os.environ["DECORD_EOF_RETRY_MAX"] = "40960"


def extract_frames(video_path: str, max_frames: int = 32, fps: float = None) -> list:
    """Extract frames from a video file.

    Args:
        video_path: Absolute path to the video file.
        max_frames: Maximum number of frames to return.
        fps: If provided, sample at this fps (capped at max_frames).
             If None, uniformly sample max_frames across the video.

    Returns:
        List of PIL.Image frames.
    """
    from decord import VideoReader, cpu

    # Use more threads for faster decoding (auto-detect or use env var)
    num_threads = int(os.environ.get("DECORD_NUM_THREADS", "8"))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=num_threads)
    total_frames = len(vr)

    if total_frames == 0:
        raise ValueError(f"Video has 0 frames: {video_path}")

    if fps is not None:
        video_fps = vr.get_avg_fps()
        step = max(1, int(video_fps / fps))
        indices = list(range(0, total_frames, step))
    else:
        num = min(max_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, num, dtype=int).tolist()

    # Cap at max_frames
    if len(indices) > max_frames:
        step = len(indices) / max_frames
        indices = [indices[int(i * step)] for i in range(max_frames)]

    frames_np = vr.get_batch(indices).asnumpy()  # (N, H, W, C)
    frames = [Image.fromarray(f) for f in frames_np]
    return frames


def extract_frames_with_metadata(
    video_path: str, max_frames: int = 32, fps: float = None
) -> tuple:
    """Extract frames and video metadata for vLLM video input.

    Returns:
        (frames_np, metadata) where frames_np is (N, H, W, 3) numpy array
        and metadata is a dict with fps, duration, total_num_frames,
        frames_indices, etc.
    """
    from decord import VideoReader, cpu

    num_threads = int(os.environ.get("DECORD_NUM_THREADS", "8"))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=num_threads)
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()
    duration = total_frames / video_fps if video_fps > 0 else 0.0

    if total_frames == 0:
        raise ValueError(f"Video has 0 frames: {video_path}")

    if fps is not None:
        step = max(1, int(video_fps / fps))
        indices = list(range(0, total_frames, step))
    else:
        num = min(max_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, num, dtype=int).tolist()

    # Cap at max_frames
    if len(indices) > max_frames:
        step = len(indices) / max_frames
        indices = [indices[int(i * step)] for i in range(max_frames)]

    frames_np = vr.get_batch(indices).asnumpy()  # (N, H, W, C)

    metadata = {
        "fps": video_fps,
        "duration": duration,
        "total_num_frames": total_frames,
        "frames_indices": indices,
        "video_backend": "decord",
        "do_sample_frames": False,
    }

    return frames_np, metadata
