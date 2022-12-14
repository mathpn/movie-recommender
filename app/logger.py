import logging
import os
from logging.config import dictConfig


LOGGER_NAME = "movie_recommender"
LOG_FORMAT = "%(levelprefix)s | %(asctime)s | %(message)s"


def create_log_config(log_level: str):
    """Logging configuration."""
    return {
        "logger_name": LOGGER_NAME,
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": LOG_FORMAT,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "movie_recommender": {"handlers": ["default"], "level": log_level},
        },
    }


dictConfig(create_log_config(log_level="DEBUG" if os.environ.get("DEBUG") else "INFO"))
logger = logging.getLogger("movie_recommender")
