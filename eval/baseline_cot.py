import logging
from .baseline import BaselineEvaluator
from benchmarks.base import MCQSample
from utils.prompt_utils import format_mcq_prompt

logger = logging.getLogger(__name__)

COT_INSTRUCTION = (
    "Let's think step by step. First, analyze the video content carefully. "
    "Then, evaluate each option against the video content. "
    "Finally, select the option that best answers the question. "
    "Conclude your response with the following format: "
    "\"The answer is [LETTER].\""
)


class BaselineCOTEvaluator(BaselineEvaluator):
    """Baseline MCQ with chain-of-thought reasoning.

    Same as baseline (original unmodified MCQ) but the model is instructed to
    reason step by step before selecting its answer.
    """

    name = "baseline_cot"

    def prepare_sample(self, sample: MCQSample) -> dict:
        prompt = format_mcq_prompt(
            sample.question, sample.options, instruction=COT_INSTRUCTION
        )
        return {
            "prompt": prompt,
            "options": sample.options,
            "expected_answer": sample.answer,
            "extra": {"num_options": len(sample.options)},
        }
