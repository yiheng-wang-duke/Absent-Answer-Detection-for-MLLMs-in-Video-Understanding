import logging
from .base import BaseEvaluator
from benchmarks.base import MCQSample
from utils.prompt_utils import format_mcq_prompt, remove_option_by_index, add_none_option

logger = logging.getLogger(__name__)

NONE_INSTRUCTION = (
    "Answer with the option's letter from the given choices directly."
)


class RemoveCorrectAddNoneEvaluator(BaseEvaluator):
    """Multi-choice detection (paper: MCDR).

    Remove correct answer + add 'None of the above' as an option.

    The correct option is removed, remaining options are re-labeled,
    and 'None of the above' is appended as the last choice.
    The model should select the 'None of the above' option.
    """

    name = "remove_correct_add_none"
    instruction = NONE_INSTRUCTION

    def prepare_sample(self, sample: MCQSample) -> dict:
        new_options, label_mapping = remove_option_by_index(
            sample.options, sample.answer_index
        )
        options = add_none_option(new_options)
        none_label = chr(ord("A") + len(options) - 1)

        prompt = format_mcq_prompt(sample.question, options, instruction=self.instruction)
        return {
            "prompt": prompt,
            "options": options,
            "expected_answer": none_label,
            "extra": {
                "removed_option": sample.options[sample.answer_index],
                "removed_answer": sample.answer,
                "label_mapping": label_mapping,
                "none_label": none_label,
                "num_options": len(options),
            },
        }
