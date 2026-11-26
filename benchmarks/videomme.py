import os
import logging
from .base import BaseBenchmark, MCQSample
from utils.prompt_utils import strip_option_labels

logger = logging.getLogger(__name__)


class VideoMMEBenchmark(BaseBenchmark):
    """VideoMME benchmark loader.

    Dataset: lmms-lab/Video-MME (2700 MCQ, 4 options A-D)
    Options come pre-labeled as ['A. text', 'B. text', ...].
    Videos are stored as {video_dir}/{videoID}.mp4.
    """

    name = "videomme"

    def load(self) -> list[MCQSample]:
        if self.anno_path and os.path.exists(self.anno_path):
            samples = self._load_from_file(self.anno_path)
        else:
            samples = self._load_from_hf()
        self.samples = samples
        logger.info(f"VideoMME: loaded {len(samples)} samples")
        return samples

    def _load_from_file(self, path: str) -> list[MCQSample]:
        import pandas as pd

        df = pd.read_parquet(path)
        return self._parse_dataframe(df)

    def _load_from_hf(self) -> list[MCQSample]:
        from datasets import load_dataset

        ds = load_dataset("lmms-lab/Video-MME", split="test")
        import pandas as pd

        df = pd.DataFrame(ds)
        return self._parse_dataframe(df)

    def _parse_dataframe(self, df) -> list[MCQSample]:
        samples = []
        for _, row in df.iterrows():
            raw_options = row["options"]
            if hasattr(raw_options, "tolist"):
                raw_options = raw_options.tolist()

            # Strip labels: "A. Apples." -> "Apples."
            options = strip_option_labels(raw_options)

            answer_letter = str(row["answer"]).strip().upper()
            answer_index = ord(answer_letter) - ord("A")

            video_id = str(row["videoID"])
            video_path = os.path.join(self.video_dir, f"{video_id}.mp4")

            samples.append(
                MCQSample(
                    sample_id=str(row["question_id"]),
                    video_path=video_path,
                    question=str(row["question"]),
                    options=options,
                    answer=answer_letter,
                    answer_index=answer_index,
                    metadata={
                        "duration": row.get("duration", ""),
                        "domain": row.get("domain", ""),
                        "sub_category": row.get("sub_category", ""),
                        "task_type": row.get("task_type", ""),
                        "videoID": video_id,
                    },
                )
            )
        return samples
