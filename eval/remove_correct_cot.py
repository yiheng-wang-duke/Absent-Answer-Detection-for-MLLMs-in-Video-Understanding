import logging
from .remove_correct import RemoveCorrectEvaluator

logger = logging.getLogger(__name__)

COT_INSTRUCTION = (
    "Let's think step by step. First, analyze the video content carefully. "
    "Then, evaluate each option against the video content. "
    "Finally, select the option that best answers the question. "
    "Conclude your response with the following format: "
    "\"The answer is [LETTER].\""
)


class RemoveCorrectCOTEvaluator(RemoveCorrectEvaluator):
    """Remove correct answer with chain-of-thought reasoning.

    Same as remove_correct (correct option removed, remaining options
    re-labeled) but the model is instructed to reason step by step before
    selecting its answer. No correct answer exists.
    """

    name = "remove_correct_cot"
    instruction = COT_INSTRUCTION
