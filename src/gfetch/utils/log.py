import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

__all__ = ["setup"]


def setup(level: int = logging.NOTSET, logfile: Path | None = None) -> None:
    """
    Configure the logging level and message format.

    Parameters
    ----------
    level : int
        Logging level for the `gfetch` logger. Defaults to `logging.NOTSET`.
    logfile : Path | None
        If given, also log to this file. Defaults to None.
    """
    fmt = "[white]%(name)s[/]\t %(message)s"

    handlers: list[logging.Handler] = [RichHandler(markup=True)]
    install()
    if logfile is not None:
        handlers.append(RichHandler(markup=True, console=Console(file=logfile.open("a+"))))
    logging.basicConfig(
        level=max(logging.INFO, level), format=fmt, datefmt="[%X]", handlers=handlers
    )
    logging.getLogger("gfetch").setLevel(level)
