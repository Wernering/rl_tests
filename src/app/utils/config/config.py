# Standard Library
import pathlib as pl


GRAPH_PATH = pl.Path("./graphs")
if not GRAPH_PATH.exists():
    GRAPH_PATH.mkdir()

LOG_PATH = pl.Path("./logs")
if not LOG_PATH.exists():
    LOG_PATH.mkdir()
