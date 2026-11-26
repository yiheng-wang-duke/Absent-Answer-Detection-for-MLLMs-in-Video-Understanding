import os
import logging
from .base import BaseBenchmark, MCQSample
from utils.prompt_utils import strip_option_labels

logger = logging.getLogger(__name__)


class EgoSchemaBenchmark(BaseBenchmark):
    """EgoSchema benchmark loader (validation subset).

    Dataset: lmms-lab/egoschema, config "Subset" (500 MCQ, 5 options A-E)
    Options come pre-labeled as ['A. text', 'B. text', ...].
    Answer is a 0-based index string (e.g., '3' means option D).
    Videos are stored as {video_dir}/{video_idx}.mp4.
    """

    name = "egoschema"

    def load(self) -> list[MCQSample]:
        if self.anno_path and os.path.exists(self.anno_path):
            samples = self._load_from_file(self.anno_path)
        else:
            samples = self._load_from_hf()
        self.samples = samples
        logger.info(f"EgoSchema: loaded {len(samples)} samples")
        return samples

    def _load_from_file(self, path: str) -> list[MCQSample]:
        import pandas as pd

        df = pd.read_parquet(path)
        return self._parse_dataframe(df)

    def _load_from_hf(self) -> list[MCQSample]:
        from datasets import load_dataset

        ds = load_dataset("lmms-lab/egoschema", "Subset", split="test")
        import pandas as pd

        df = pd.DataFrame(ds)
        return self._parse_dataframe(df)

    def _parse_dataframe(self, df) -> list[MCQSample]:
        samples = []
        for _, row in df.iterrows():
            raw_options = row["option"]
            if hasattr(raw_options, "tolist"):
                raw_options = raw_options.tolist()

            # Strip labels: "A. text" -> "text"
            options = strip_option_labels(raw_options)

            # Answer is a 0-based index string
            answer_index = int(row["answer"])
            answer_letter = chr(ord("A") + answer_index)

            video_id = str(row["video_idx"])
            video_path = os.path.join(self.video_dir, f"{video_id}.mp4")

            samples.append(
                MCQSample(
                    sample_id=str(row["question_idx"]),
                    video_path=video_path,
                    question=str(row["question"]),
                    options=options,
                    answer=answer_letter,
                    answer_index=answer_index,
                    metadata={
                        "video_idx": video_id,
                    },
                )
            )
        return samples
