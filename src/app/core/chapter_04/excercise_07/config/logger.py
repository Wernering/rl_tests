# External
from config.config import LOG_PATH
from pydantic import BaseSettings


LOG_NAME = "logger"
LOGGER_LEVEL = "DEBUG"


class LoggerConfig(BaseSettings):
    version = 1

    formatters = {
        "std": {
            "format": "%(asctime)s-%(levelname)s-%(name)s::%(module)s|%(lineno)s:: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    }

    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "std",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "std",
            "level": "DEBUG",
            "filename": LOG_PATH,
            "mode": "a",
            "maxBytes": 1048576,
            "backupCount": 10,
        },
    }

    loggers = {
        LOG_NAME: {
            "level": LOGGER_LEVEL,
            "handlers": ["console", "file"],
            "propagate": False,
        },
    }
