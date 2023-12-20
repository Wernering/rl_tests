# External
import matplotlib.pyplot as plt
import numpy as np

# Project
from app.utils import save_graph

# Local
from ..config import FILE_NAME, LOGGER


def get_value_as_matrix(value: tuple | np.ndarray) -> np.ndarray:
    if isinstance(value, tuple):
        value = np.array(*value)
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
        ax.plot(results.keys(), results.values(), label=f"planning steps: {planning_steps}")

    ax.legend(loc="upper right")

    graph_name = f"dynaMaze_rngSeed_{seed}_alpha_{alpha}_gamma_{gamma}_epsilon_{epsilon}.png"
    save_graph(fig, name=graph_name, file_name=FILE_NAME, logger=LOGGER, date_id=False)
