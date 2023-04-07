import os

import numpy as np
import matplotlib.pyplot as plt

import config.config as config


def get_value_as_matrix(value: np.array) -> np.array:
    return value + [1, 1]


def graph(
    cycle_results: dict[int, dict[int, int]],
    seed,
    alpha: float,
    gamma: float,
    epsilon: float,
) -> None:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title("Dyna Maze")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Steps per pisodes")

    for planning_steps, results in cycle_results.items():
        ax.plot(
            results.keys(), results.values(), label=f"planning steps: {planning_steps}"
        )

    ax.legend(loc="upper right")

    if not os.path.isdir(config.GRAPH_PATH):
        os.mkdir(config.GRAPH_PATH)

    plt.savefig(
        f"{config.GRAPH_PATH}/dynaMaze_rngSeed_{seed}_alpha_{alpha}_gamma_{gamma}_epsilon_{epsilon}.png"
    )
