# External
import matplotlib.pyplot as plt
import numpy as np

# Project
from app.utils import save_graph

# Local
from ..classes.problem import KBanditProblem
from ..config import FILE_NAME, LOGGER


def episode(problem: KBanditProblem, iterations: int) -> list:
    for _ in range(iterations + 1):
        problem.iteration()
    return problem.results


def cycle(problem: KBanditProblem, episode_iterations: int, cycle_iterations: int) -> dict:
    results = {}
    for i in range(cycle_iterations + 1):
        problem.reset_bandits()
        results[i] = episode(problem=problem, iterations=episode_iterations)

    return results


def average_result(result: dict[int, list]) -> list:
    arr = np.array(list(result.values()))
    return np.mean(arr, axis=0)


def graph(cycle_results: dict[int, list], episodes: int, cycles: int) -> None:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title("K-Armed Bandit")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Average Reward")

    for epsilon, results in cycle_results.items():
        ax.plot(range(len(results)), results, label=rf"$\epsilon$-{epsilon}")

    ax.legend(loc="lower right")

    graph_name = f"k_armed_bandit_cycles_{cycles}_episodes_{episodes}"
    save_graph(fig=fig, name=graph_name, file_name=FILE_NAME, logger=LOGGER, date_id=False)
