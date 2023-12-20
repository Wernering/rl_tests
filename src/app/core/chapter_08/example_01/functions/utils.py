# External
import matplotlib.pyplot as plt
import numpy as np

# Project
from app.utils import GRAPH_PATH

# Local
from ..config import FILE_NAME, LOGGER


LOCAL_GRAPH_PATH = GRAPH_PATH.joinpath(FILE_NAME)


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
    _, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title("Dyna Maze")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Steps per pisodes")

    for planning_steps, results in cycle_results.items():
        ax.plot(results.keys(), results.values(), label=f"planning steps: {planning_steps}")

    ax.legend(loc="upper right")

    if not LOCAL_GRAPH_PATH.exists():
        LOCAL_GRAPH_PATH.mkdir()

    LOGGER.info(f"Saving image in {LOCAL_GRAPH_PATH}")
    plt.savefig(LOCAL_GRAPH_PATH.joinpath(f"dynaMaze_rngSeed_{seed}_alpha_{alpha}_gamma_{gamma}_epsilon_{epsilon}.png"))
