import os
import platform
from statistics import mean

import matplotlib.pyplot as plt

import config.config as config
from classes.problem import KBanditProblem


def episode(problem: KBanditProblem, iterations: int) -> list:
    for _ in range(iterations + 1):
        problem.iteration()
    return problem.results


def cycle(
    problem: KBanditProblem, episode_iterations: int, cycle_iterations: int
) -> dict:
    results = {}
    for i in range(cycle_iterations + 1):
        problem.reset_bandits()
        results[i] = episode(problem=problem, iterations=episode_iterations)

    return results


def average_result(result: dict[int, list]) -> list:
    return [mean(x for x in t) for t in zip(*list(result.values()))]


def graph(
    cycle_results: dict[int, list], episodes: int, cycles: int, alpha: float
) -> None:
    _, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title("K-Armed Bandit")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Average Reward")

    for epsilon, results in cycle_results.items():
        ax.plot(range(len(results)), results, label=rf"$\epsilon$-{epsilon}")

    ax.legend(loc="lower right")

    if not os.path.isdir(config.GRAPH_PATH):
        os.mkdir(config.GRAPH_PATH)

    path = f"{config.GRAPH_PATH}/"
    if platform.system() == "Windows":
        path = f"{config.GRAPH_PATH}\\"

    plt.savefig(
        f"{path}k_armed_bandit_cycles_{cycles}_episodes_{episodes}_alpha_{alpha}.png"
    )
