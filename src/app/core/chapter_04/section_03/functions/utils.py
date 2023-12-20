# External
import matplotlib.pyplot as plt
import numpy as np

# Project
from app.utils import GRAPH_PATH

# Local
from ..classes.rental import JacksRental
from ..config import FILE_NAME, LOGGER


LOCAL_GRAPH_PATH = GRAPH_PATH.joinpath(FILE_NAME)


def save_graph(name: str, id) -> None:
    if not LOCAL_GRAPH_PATH.exists():
        LOCAL_GRAPH_PATH.mkdir()
    LOGGER.info(f"Saving image '{name}_{id}.png' in {LOCAL_GRAPH_PATH}")
    plt.savefig(LOCAL_GRAPH_PATH.joinpath(f"{name}_{id}.png"))


def graph_state_value(solution: JacksRental, id) -> None:
    matrix = solution.state_value
    matrix = np.fliplr(matrix)

    fig = plt.figure()
    fig.suptitle(f"Loc.A: {solution.location_a}\nLoc.B: {solution.location_b}", fontsize=12)
    ax = fig.add_subplot(projection="3d")

    x_ax, y_ax = matrix.shape
    X, Y = np.meshgrid(np.arange(0, x_ax)[::-1], np.arange(0, y_ax))

    ax.plot_surface(
        X,
        Y,
        matrix,
        edgecolor="royalblue",
        lw=0.5,
        rstride=8,
        cstride=8,
        alpha=0.3,
    )

    min_val = np.amin(matrix)
    max_val = np.amax(matrix)
    ax.set(
        zlim=(min_val, max_val),
        xlabel="Location A",
        ylabel="Location B",
        zlabel="Expected value",
    )

    save_graph("state_value", id)


def graph_policy_value(solution: JacksRental, id) -> None:
    matrix = solution.policy_matrix
    moves = list(np.arange(-solution.max_move, solution.max_move + 1, dtype=float))

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(
        f"Loc.A: {solution.location_a}\nLoc.B: {solution.location_b}",
        fontsize=10,
        y=0.98,
    )

    x_ax, y_ax = matrix.shape
    x_range = np.arange(0, x_ax)
    y_range = np.arange(0, y_ax)[::-1]

    cmap = plt.get_cmap("RdBu", 2 * solution.max_move + 1)  # type: ignore
    cm = ax.pcolormesh(matrix, cmap=cmap, vmin=-solution.max_move - 0.5, vmax=solution.max_move + 0.5)
    ax.set_xticks(x_range)
    ax.set_yticks(y_range)

    fig.colorbar(cm, ticks=moves)
    fig.tight_layout()

    save_graph("policy_value", id)
