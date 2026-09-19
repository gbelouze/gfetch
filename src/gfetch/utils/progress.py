import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

__all__ = ["default_bar", "gfetch_debug", "temporary_task"]

log = logging.getLogger(__name__)


def gfetch_debug() -> bool:
    """
    Whether progress bars should be forced off via the `GFETCH_DEBUG` env var.

    Returns
    -------
    bool
        True if `GFETCH_DEBUG` is set to `1` or `true`.
    """
    return os.getenv("GFETCH_DEBUG") in ("1", "true")


def default_bar() -> Progress:
    """
    Build the shared `rich.progress.Progress` instance used across gfetch commands.

    Disabled when `GFETCH_DEBUG` is set, or when stdout isn't attached to a terminal
    (e.g. a SLURM batch job's captured stdout/stderr), so batch logs aren't spammed
    with animated-bar redraws.

    Returns
    -------
    Progress
        A `rich.progress.Progress` instance, built once per CLI command and passed
        down to whatever needs to report progress.
    """
    disabled = gfetch_debug() or not Console().is_terminal
    if disabled:
        log.debug("Progress bar is disabled.")
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        refresh_per_second=1,
        disable=disabled,
    )


@contextmanager
def temporary_task(progress: Progress, *args: Any, **kwargs: Any) -> Iterator[TaskID]:
    """
    Add a progress task and ensure it is removed when the context exits.

    Parameters
    ----------
    progress : Progress
        A `rich.progress.Progress` instance managing the tasks.
    *args : Any
        Positional arguments passed to `progress.add_task`.
    **kwargs : Any
        Keyword arguments passed to `progress.add_task`.

    Yields
    ------
    TaskID
        The task ID returned by `progress.add_task`.
    """
    task = None
    try:
        task = progress.add_task(*args, **kwargs)
        yield task
    finally:
        if task is not None:
            progress.update(task, visible=False)
            progress.remove_task(task)
