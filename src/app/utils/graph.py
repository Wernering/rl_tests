# Standard Library
import datetime as dt
import logging

# External
from matplotlib.figure import Figure

# Local
from .config import GRAPH_PATH


def save_graph(
    fig: Figure,
    name: str,
    file_name: str,
    logger: logging.Logger,
    date_id: bool = True,
    img_fmt: str = "png",
) -> None:
    if date_id:
        idt = dt.date.today()
        name += f"_{idt}"

    if img_fmt is not None:
        name += f".{img_fmt}"

    graph_path = GRAPH_PATH.joinpath(file_name)
    if not graph_path.exists():
        graph_path.mkdir()

    logger.info(f"Saving image '{name}' in {graph_path}")
    fig.savefig(graph_path.joinpath(f"{name}"))
