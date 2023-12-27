# External
import matplotlib.pyplot as plt

# Project
from app.utils import save_graph

# Local
from ..config import FILE_NAME, LOGGER


def graph(cycle_results: tuple[str, list], episodes: int, cycles: int) -> None:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title("Q-Learning vs Double Q-Learning")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("% left actions from A")

    for name, results in cycle_results:
        ax.plot(range(len(results)), results, label=name)

    ax.legend(loc="upper right")

    graph_name = f"k_armed_bandit_cycles_{cycles}_episodes_{episodes}"
    save_graph(fig=fig, name=graph_name, file_name=FILE_NAME, logger=LOGGER, date_id=False)
