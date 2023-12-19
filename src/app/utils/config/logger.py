# Standard Library
import logging
from logging.config import dictConfig

# Local
from .config import LOG_PATH


LOGGER_NAME = "default_log"

LOG_FORMAT: str = (
    '{"time": "%(asctime)s", "level": "%(levelname)s", '
    + '"thread": "%(threadName)s", "component": "%(module)s",'
    + f'"payload": %(message)s}}'
)


def get_config(log_name: str) -> dict:
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "class": "logging.Formatter",
                "format": LOG_FORMAT,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": logging.INFO,
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "default",
                "level": logging.INFO,
                "filename": LOG_PATH.joinpath(f"{log_name}.log"),
                "mode": "a",
                "encoding": "utf-8",
                "maxBytes": 5000000,
                "backupCount": 4,
            },
            "debug_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "default",
                "level": logging.DEBUG,
                "filename": LOG_PATH.joinpath(f"{log_name}_debug.log"),
                "mode": "a",
                "encoding": "utf-8",
                "maxBytes": 5000000,
                "backupCount": 4,
            },
        },
        "loggers": {
            log_name: {
                "handlers": ["console", "file", "debug_file"],
                "level": logging.DEBUG,
                "propagate": False,
            },
        },
    }


def create_logger(config: dict = None, file_name: str = LOGGER_NAME) -> logging.Logger:
    if config is None:
        config = get_config(file_name)

    dictConfig(config)
    return logging.getLogger(file_name)
