import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from dask.callbacks import Callback
from rich.console import Console, RenderableType
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.text import Text

__all__ = ["count_bar", "dask_progress", "default_bar", "gfetch_debug", "temporary_task"]

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


def _disabled() -> bool:
    """
    Whether a `Progress` instance should render disabled.

    Disabled when `GFETCH_DEBUG` is set, or when stdout isn't attached to a terminal
    (e.g. a SLURM batch job's captured stdout/stderr), so batch logs aren't spammed
    with animated-bar redraws.

    Returns
    -------
    bool
        True if progress bars should be disabled.
    """
    disabled = gfetch_debug() or not Console().is_terminal
    if disabled:
        log.debug("Progress bar is disabled.")
    return disabled


class _PerUnitColumn(ProgressColumn):
    """
    Column rendering differently for byte-counting and item-counting tasks.

    A task counts bytes when it was added with a `bytes=True` field, e.g.
    `progress.add_task(description, bytes=True)`.
    """

    def __init__(
        self, count_column: ProgressColumn | None, bytes_column: ProgressColumn | None
    ) -> None:
        super().__init__()
        self._count_column = count_column
        self._bytes_column = bytes_column

    def render(self, task: Task) -> RenderableType:
        column = self._bytes_column if task.fields.get("bytes", False) else self._count_column
        return column(task) if column is not None else Text("")


def default_bar() -> Progress:
    """
    Build the shared `rich.progress.Progress` instance used for byte-oriented
    progress (e.g. downloads) across gfetch commands.

    Tasks added with a `bytes=True` field show sizes and a transfer speed, the
    others an item count.

    Returns
    -------
    Progress
        A `rich.progress.Progress` instance, built once per CLI command and passed
        down to whatever needs to report progress.
    """
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        _PerUnitColumn(MofNCompleteColumn(), DownloadColumn()),
        _PerUnitColumn(None, TransferSpeedColumn()),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        refresh_per_second=1,
        disable=_disabled(),
    )


def count_bar() -> Progress:
    """
    Build the shared `rich.progress.Progress` instance used for count-oriented
    progress (e.g. zones, dask task completion) across gfetch commands.

    Returns
    -------
    Progress
        A `rich.progress.Progress` instance, built once per CLI command and passed
        down to whatever needs to report progress.
    """
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        refresh_per_second=1,
        disable=_disabled(),
    )


class _RichDaskCallback(Callback):
    """
    Dask scheduler callback that drives a `rich.progress.Progress` task.

    Reports per-task completion of one `.compute()`/`.load()` call on `progress`, as
    a stand-in for `dask.diagnostics.ProgressBar()`'s own separate text bar - letting
    a dask computation's progress live in the same `Progress` instance (and terminal
    region) as any other bar the caller is tracking.
    """

    def __init__(self, progress: Progress, task: TaskID) -> None:
        self._progress = progress
        self._task = task

    def _start_state(self, dsk: Any, state: dict[str, Any]) -> None:
        total = len(state["finished"]) + sum(len(state[k]) for k in ("ready", "waiting", "running"))
        self._progress.update(self._task, total=total)

    def _posttask(
        self, key: Any, result: Any, dsk: Any, state: dict[str, Any], worker_id: Any
    ) -> None:
        self._progress.advance(self._task)


@contextmanager
def dask_progress(progress: Progress, description: str) -> Iterator[None]:
    """
    Track one dask-backed `.compute()`/`.load()` call's tasks on `progress`.

    Adds a temporary task to `progress` for the duration of the `with` block,
    advanced once per completed dask task, then removes it.

    Parameters
    ----------
    progress : Progress
        The shared `Progress` instance to add a task to.
    description : str
        Description shown next to this call's task-completion bar.

    Yields
    ------
    None
    """
    with (
        temporary_task(progress, description, total=None) as task,
        _RichDaskCallback(progress, task),
    ):
        yield


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
