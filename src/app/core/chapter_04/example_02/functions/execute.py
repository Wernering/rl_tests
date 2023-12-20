# External
import numpy as np

# Project
from app.utils import ctx_timer

# Local
from ..classes.problem import GamblersProblem
from ..config import LOGGER


def one_step_lookahead(problem: GamblersProblem, old_vf: np.ndarray) -> GamblersProblem:
    for state in range(1, problem.get_states()):
        expected_value = {}

        for action in range(1, min(state, 100 - state) + 1):
            expected_value[action] = problem.calculate_expected_return(old_vf, state, action)
        max_value = max(expected_value.values())
        best_action = min([a for a, v in expected_value.items() if v == max_value])
        problem.update_vf(state=state, new_value=max_value)
        problem.update_policy(state=state, new_action=best_action)

    return problem


def execute(problem: GamblersProblem, theta: float) -> GamblersProblem:
    theta = theta
    delta = np.inf
    it = 0

    while delta > theta:
        it += 1
        with ctx_timer(f"Iteration {it}", logger=LOGGER):
            old_vf = problem.vf.copy()
            problem = one_step_lookahead(problem, old_vf=old_vf)
            delta = np.amax(abs(problem.vf - old_vf))

            LOGGER.info(f"Theta Value: {theta}. Max difference: {delta}")

    return problem
