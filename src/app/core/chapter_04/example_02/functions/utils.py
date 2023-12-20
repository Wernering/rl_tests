# External
import matplotlib.pyplot as plt

# Project
from app.utils import GRAPH_PATH

# Local
from ..classes.problem import GamblersProblem
from ..config import FILE_NAME, LOGGER


LOCAL_GRAPH_PATH = GRAPH_PATH.joinpath(FILE_NAME)


def save_graph(name: str, id) -> None:
    if not LOCAL_GRAPH_PATH.exists():
        LOCAL_GRAPH_PATH.mkdir()
    LOGGER.info(f"Saving image '{name}_{id}.png' in {LOCAL_GRAPH_PATH}")
    plt.savefig(LOCAL_GRAPH_PATH.joinpath(f"{name}_{id}.png"))


def graph_state_value(solution: GamblersProblem, id, theta: float) -> None:
    matrix = solution.vf[1:-1]

    f, ax = plt.subplots(figsize=(9, 6))
    f.suptitle(rf"Value Function. ph {solution.ph}. " r"$\theta$ " f"{theta}")
    ax.set(xlabel="Capital", ylabel="Value Estimates")
    ax.plot(range(1, solution.get_states()), matrix)

    save_graph(f"state_value_ph_{solution.ph}_theta_{theta}", id=id)


def graph_policy_value(solution: GamblersProblem, id, theta: float) -> None:
    matrix = solution.policy[1:-1]

    f, ax = plt.subplots(figsize=(9, 6))
    f.suptitle(rf"Policy Function. ph {solution.ph}. " r"$\theta$ " f"{theta}")
    ax.set(xlabel="Capital", ylabel="Policy")
    ax.bar(range(1, solution.get_states()), matrix)

    save_graph(f"policy_value_ph_{solution.ph}_theta_{theta}", id=id)
