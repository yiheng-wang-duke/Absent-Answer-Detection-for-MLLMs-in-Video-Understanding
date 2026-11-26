import logging
from .base import BaseEvaluator
from benchmarks.base import MCQSample
from utils.prompt_utils import format_mcq_prompt

logger = logging.getLogger(__name__)


class BaselineEvaluator(BaseEvaluator):
    """Baseline (paper: ACC): the original, unmodified MCQ."""

    name = "baseline"

    def prepare_sample(self, sample: MCQSample) -> dict:
        prompt = format_mcq_prompt(sample.question, sample.options)
        return {
            "prompt": prompt,
            "options": sample.options,
            "expected_answer": sample.answer,
            "extra": {"num_options": len(sample.options)},
        }
