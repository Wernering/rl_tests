# External
import matplotlib.pyplot as plt

# Project
from app.utils import save_graph

# Local
from ..config import FILE_NAME, LOGGER


def plot_estimated_value(series: dict[str, list], name, id):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title("Estimated Value")
    ax.set_xlabel("State")
    ax.set_ylabel("Estimated Value")

    for label, serie in series.items():
        ax.plot(range(len(serie)), serie, label=label)

    ax.legend(loc="lower right")

    save_graph(fig=fig, name=name, file_name=FILE_NAME, logger=LOGGER)


def plot_rms(series: dict[str, list], name, id):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title("RMS")
    ax.set_xlabel("walks / Episodes")
    ax.set_ylabel("rms")

    for label, serie in series.items():
        ax.plot(range(len(serie)), serie, label=label)

    ax.legend(loc="upper right")

    save_graph(fig=fig, name=name, file_name=FILE_NAME, logger=LOGGER)
