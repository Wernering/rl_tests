from pydantic import BaseSettings
from config.config import LOG_PATH

LOG_NAME = "logger"
LOGGER_LEVEL = "INFO"


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
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "std",
            "level": "INFO",
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
