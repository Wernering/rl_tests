import logging
import time

from config.logger import LOG_NAME
from classes.problem import KBanditProblem
from functions.utils import cycle, average_result

logger = logging.getLogger(LOG_NAME)


def execute(
    epsilons: list[float],
    k: int,
    episodes: int,
    cycles: int,
    initial_guess: float = 0,
    alpha: float = 0.5,
) -> dict:
    problems = []

    base = KBanditProblem(
        k=k,
        initial_guess=initial_guess,
        alpha=alpha,
    )

    for e in epsilons:
        p = base.copy()
        p.change_epsilon(e)
        problems.append(p)

    experiment_result = {}
    for i, problem in enumerate(problems, 1):
        logger.info(f"Executing problem {i}, with epsilon: {problem.get_epsilon()}")
        ep_time_init = time.time()
        results = cycle(
            problem=problem, episode_iterations=episodes, cycle_iterations=cycles
        )
        results = average_result(results)
        logger.info(f"Execution Time: {time.time() - ep_time_init} seconds")
        experiment_result[problem.get_epsilon()] = results

    return experiment_result
