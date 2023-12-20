# Project
from app.utils import ctx_timer

# Local
from ..classes.problem import KBanditProblem
from ..config import LOGGER
from .utils import average_result, cycle


def execute(
    epsilons: list[float],
    k: int,
    episodes: int,
    cycles: int,
    initial_guess: float = 0,
) -> dict:
    problems: list[KBanditProblem] = []

    base = KBanditProblem(k=k, initial_guess=initial_guess)

    for e in epsilons:
        p = base.copy()
        p.change_epsilon(e)
        problems.append(p)

    experiment_result = {}
    for i, problem in enumerate(problems, 1):
        LOGGER.info(f"Executing problem {i}, with epsilon: {problem.get_epsilon()}")
        with ctx_timer("Whole Cycle", logger=LOGGER):
            results = cycle(problem=problem, episode_iterations=episodes, cycle_iterations=cycles)
        LOGGER.info("Calculating Average return per step over cycles")
        results = average_result(results)
        experiment_result[problem.get_epsilon()] = results

    return experiment_result
