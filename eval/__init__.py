from .baseline import BaselineEvaluator
from .baseline_cot import BaselineCOTEvaluator
from .remove_correct import RemoveCorrectEvaluator
from .remove_correct_cot import RemoveCorrectCOTEvaluator
from .remove_correct_add_none import RemoveCorrectAddNoneEvaluator
from .remove_correct_add_none_cot import RemoveCorrectAddNoneCOTEvaluator
from .remove_correct_force_none import RemoveCorrectForceNoneEvaluator
from .remove_correct_force_none_cot import RemoveCorrectForceNoneCOTEvaluator

EVALUATORS = {
    "baseline": BaselineEvaluator,
    "baseline_cot": BaselineCOTEvaluator,
    "remove_correct": RemoveCorrectEvaluator,
    "remove_correct_cot": RemoveCorrectCOTEvaluator,
    "remove_correct_add_none": RemoveCorrectAddNoneEvaluator,
    "remove_correct_add_none_cot": RemoveCorrectAddNoneCOTEvaluator,
    "remove_correct_force_none": RemoveCorrectForceNoneEvaluator,
    "remove_correct_force_none_cot": RemoveCorrectForceNoneCOTEvaluator,
}
