import logging

import typer
from rich.logging import RichHandler

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            tracebacks_suppress=[typer],
        )
    ],
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
