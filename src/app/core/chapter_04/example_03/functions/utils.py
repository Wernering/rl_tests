# Standard Library
import logging
import os
import platform
import time
from contextlib import contextmanager

# External
import config.config as config
import matplotlib.pyplot as plt
from classes.problem import GamblersProblem
from config.logger import LOG_NAME


logger = logging.getLogger(LOG_NAME)


@contextmanager
def timer(label: str):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        logger.info(f"{label}: total execution time: {end_time - start_time:.3f} seconds")


def save_graph(name: str, id) -> None:
    if not os.path.isdir(config.GRAPH_PATH):
        os.mkdir(config.GRAPH_PATH)

    path = f"{config.GRAPH_PATH}/"
    if platform.system() == "Windows":
        path = f"{config.GRAPH_PATH}\\"

    plt.savefig(f"{path}{name}_{id}.png")


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
