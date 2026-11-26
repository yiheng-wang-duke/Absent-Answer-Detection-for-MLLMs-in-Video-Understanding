import logging
from .remove_correct_add_none import RemoveCorrectAddNoneEvaluator

logger = logging.getLogger(__name__)

COT_NONE_INSTRUCTION = (
    "Let's think step by step. First, analyze the video content carefully. "
    "Then, evaluate each option against the video content. "
    "Finally, assess whether any option fully and correctly answers the "
    "question based on the video. If no other option is correct, select "
    "\"None of the above.\" Conclude your response with the following format: "
    "\"The answer is [LETTER].\""
)


class RemoveCorrectAddNoneCOTEvaluator(RemoveCorrectAddNoneEvaluator):
    """Remove correct answer + add 'None of the above' with chain-of-thought.

    Same as remove_correct_add_none but the model is instructed to reason
    step by step before selecting its answer.
    """

    name = "remove_correct_add_none_cot"
    instruction = COT_NONE_INSTRUCTION
