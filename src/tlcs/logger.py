import logging

import typer
from rich.logging import RichHandler


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger to use RichHandler.

    Args:
        level: Logging level to configure for the root logger.
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                tracebacks_suppress=[typer],
            )
        ],
    )


configure_logging()


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with the global Rich logging setup.

    Args:
        name: Name of the logger (usually ``__name__``).

    Returns:
        The logger instance with the given name.
    """
    return logging.getLogger(name)
