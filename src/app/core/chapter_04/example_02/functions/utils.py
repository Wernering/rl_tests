# External
import matplotlib.pyplot as plt

# Project
from app.utils import save_graph

# Local
from ..classes.problem import GamblersProblem
from ..config import FILE_NAME, LOGGER


def graph_state_value(solution: GamblersProblem, theta: float) -> None:
    matrix = solution.vf[1:-1]

    f, ax = plt.subplots(figsize=(9, 6))
    f.suptitle(rf"Value Function. ph {solution.ph}. " r"$\theta$ " f"{theta}")
    ax.set(xlabel="Capital", ylabel="Value Estimates")
    ax.plot(range(1, solution.get_states()), matrix)

    graph_name = f"state_value_ph_{solution.ph}_theta_{theta}"
    save_graph(fig=f, name=graph_name, file_name=FILE_NAME, logger=LOGGER)


def graph_policy_value(solution: GamblersProblem, theta: float) -> None:
    matrix = solution.policy[1:-1]

    f, ax = plt.subplots(figsize=(9, 6))
    f.suptitle(rf"Policy Function. ph {solution.ph}. " r"$\theta$ " f"{theta}")
    ax.set(xlabel="Capital", ylabel="Policy")
    ax.bar(range(1, solution.get_states()), matrix)

    graph_name = f"policy_value_ph_{solution.ph}_theta_{theta}"
    save_graph(fig=f, name=graph_name, file_name=FILE_NAME, logger=LOGGER)
