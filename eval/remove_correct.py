import logging
from .base import BaseEvaluator
from benchmarks.base import MCQSample
from utils.prompt_utils import format_mcq_prompt, remove_option_by_index

logger = logging.getLogger(__name__)


class RemoveCorrectEvaluator(BaseEvaluator):
    """Unprompted detection (paper: UDR).

    Removes the correct option from the MCQ and re-labels the remaining
    options, under the standard instruction (no detection cue).
    """

    name = "remove_correct"
    instruction = "Answer with the option's letter from the given choices directly."

    def prepare_sample(self, sample: MCQSample) -> dict:
        new_options, label_mapping = remove_option_by_index(
            sample.options, sample.answer_index
        )
        prompt = format_mcq_prompt(
            sample.question,
            new_options,
            instruction=self.instruction,
        )
        return {
            "prompt": prompt,
            "options": new_options,
            "expected_answer": None,  # No correct answer exists
            "extra": {
                "removed_option": sample.options[sample.answer_index],
                "removed_answer": sample.answer,
                "label_mapping": label_mapping,
                "num_options": len(new_options),
            },
        }
