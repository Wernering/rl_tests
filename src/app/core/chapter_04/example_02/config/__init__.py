# Project
from app.utils import create_logger

# Local
from .config import FILE_NAME, STATES


LOGGER = create_logger(file_name=FILE_NAME)

__all__ = [
    "FILE_NAME",
    "LOGGER",
    "STATES",
]
