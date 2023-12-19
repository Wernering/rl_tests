# Standard Library
import logging
import os
import platform
import time
from contextlib import contextmanager

# External
import matplotlib.pyplot as plt
from config import config
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


def plot_estimated_value(series: dict[str, list], name, id):
    _, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title("Estimated Value")
    ax.set_xlabel("State")
    ax.set_ylabel("Estimated Value")

    for label, serie in series.items():
        ax.plot(range(len(serie)), serie, label=label)

    ax.legend(loc="lower right")

    save_graph(name=name, id=id)


def plot_rms(series: dict[str, list], name, id):
    _, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title("RMS")
    ax.set_xlabel("walks / Episodes")
    ax.set_ylabel("rms")

    for label, serie in series.items():
        ax.plot(range(len(serie)), serie, label=label)

    ax.legend(loc="upper right")

    save_graph(name=name, id=id)
