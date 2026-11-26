import logging
from .remove_correct_force_none import RemoveCorrectForceNoneEvaluator

logger = logging.getLogger(__name__)

COT_FORCE_NONE_INSTRUCTION = (
    "Let's think step by step. First, analyze the video content carefully. "
    "Then, evaluate each option against the video content. "
    "Finally, assess whether any option fully and correctly answers the "
    "question. If none of the options is correct, you should respond with "
    "\"The answer is None of the above.\" Otherwise, conclude with "
    "\"The answer is [LETTER].\""
)


class RemoveCorrectForceNoneCOTEvaluator(RemoveCorrectForceNoneEvaluator):
    """Remove correct answer + instruct 'None of the above' with chain-of-thought.

    Same as remove_correct_force_none but the model is instructed to reason
    step by step before selecting its answer. 'None of the above' is NOT
    listed as a selectable option — it's only mentioned in the instruction.
    """

    name = "remove_correct_force_none_cot"
    instruction = COT_FORCE_NONE_INSTRUCTION
