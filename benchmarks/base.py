from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class MCQSample:
    """Uniform representation of a single MCQ sample across all benchmarks."""

    sample_id: str
    video_path: str
    question: str
    options: list[str]      # Raw option text WITHOUT letter labels
    answer: str             # Correct answer letter (A, B, C, ...)
    answer_index: int       # 0-based index of correct answer in options
    metadata: dict = field(default_factory=dict)


class BaseBenchmark(ABC):
    """Abstract base for benchmark data loading."""

    name: str = ""

    def __init__(self, video_dir: str, anno_path: str = None, split: str = "test"):
        self.video_dir = video_dir
        self.anno_path = anno_path
        self.split = split
        self.samples: list[MCQSample] = []

    @abstractmethod
    def load(self) -> list[MCQSample]:
        """Load all samples and store them in self.samples."""
        ...

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MCQSample:
        return self.samples[idx]
