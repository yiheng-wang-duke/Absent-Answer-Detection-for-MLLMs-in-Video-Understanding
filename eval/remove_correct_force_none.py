import logging
from .base import BaseEvaluator
from benchmarks.base import MCQSample
from utils.prompt_utils import format_mcq_prompt, remove_option_by_index

logger = logging.getLogger(__name__)

FORCE_NONE_INSTRUCTION = (
    "If none of the given options are correct, output 'None of the above'. "
    "Answer with the option's letter or 'None of the above'."
)


class RemoveCorrectForceNoneEvaluator(BaseEvaluator):
    """Open-ended detection (paper: OEDR).

    Remove correct answer + instruct model to output 'None of the above'.

    The correct option is removed and the model is explicitly instructed
    to output 'None of the above' when no option is correct.
    Unlike remove_correct_add_none, 'None of the above' is NOT listed
    as a selectable option — it's only mentioned in the instruction.
    """

    name = "remove_correct_force_none"
    instruction = FORCE_NONE_INSTRUCTION

    def prepare_sample(self, sample: MCQSample) -> dict:
        new_options, label_mapping = remove_option_by_index(
            sample.options, sample.answer_index
        )
        prompt = format_mcq_prompt(
            sample.question, new_options, instruction=self.instruction
        )
        return {
            "prompt": prompt,
            "options": new_options,
            "expected_answer": "None of the above",
            "extra": {
                "removed_option": sample.options[sample.answer_index],
                "removed_answer": sample.answer,
                "label_mapping": label_mapping,
                "num_options": len(new_options),
            },
        }
