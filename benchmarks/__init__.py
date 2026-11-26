from .videomme import VideoMMEBenchmark
from .egoschema import EgoSchemaBenchmark

BENCHMARKS = {
    "videomme": VideoMMEBenchmark,
    "egoschema": EgoSchemaBenchmark,
}


def build_benchmark(name: str, **kwargs):
    """Build a benchmark by name.

    Args:
        name: One of 'videomme', 'egoschema'.
        **kwargs: Passed to the benchmark constructor (video_dir, anno_path, split).
    """
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name}. Choose from {list(BENCHMARKS.keys())}")
    return BENCHMARKS[name](**kwargs)
