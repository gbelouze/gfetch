"""Shared constants, search helpers, logging and progress-bar setup for the
benchmark scripts in this directory. No dependency on the gfetch package itself.
"""

import logging
import multiprocessing
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Literal

import pystac
from dask.callbacks import Callback
from pystac_client import Client
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
BANDS = ["red", "green", "blue"]
RESOLUTION = 10

# "wide" and "deep" scenarios from https://benchmark-odc-stac-vs-stackstac.netlify.app/
WIDE_BBOX = (27.345815, -14.98724, 27.565542, -7.710992)
WIDE_DATETIME = "2020-06-06"
WIDE_CRS = "EPSG:32735"  # UTM 35S - matches the MGRS tiles covered by WIDE_BBOX

DEEP_BBOX = (27.4, -8.2, 27.5, -7.8)
DEEP_DATETIME = "2020-06-01/2020-07-31"
DEEP_TILE = "MGRS-35MNM"
DEEP_CRS = "EPSG:32735"

Scenario = Literal["wide", "deep"]

log = logging.getLogger(__name__)

# Extra time given to a subprocess case beyond its own `max_time` before
# `run_case_in_subprocess()` gives up waiting and kills it outright - covers
# setup/search/graph-building time outside whatever section the case itself
# times internally.
SAFETY_GRACE_S = 30.0


def setup_logging(verbose: bool = False) -> None:
    """Configure rich-formatted logging shared across the benchmark scripts.

    Parameters
    ----------
    verbose : bool
        Log at `DEBUG` instead of `INFO` when True. Defaults to False.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(name)s: %(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(markup=False, rich_tracebacks=True)],
        force=True,
    )


def sweep_progress() -> Progress:
    """Build the shared `rich.progress.Progress` instance for a benchmark sweep.

    One task per sweep (e.g. one per script, or one per stage in a multi-stage
    script) tracks how many of the sweep's test configurations have completed,
    alongside elapsed/remaining time - orthogonal to any per-run logging of that
    configuration's own throughput.

    Returns
    -------
    Progress
        A `rich.progress.Progress` instance, built once per benchmark run and
        passed down to whatever needs to report sweep progress.
    """
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


class _RichDaskCallback(Callback):
    """Dask scheduler callback that drives a `rich.progress.Progress` task.

    Reports per-task completion of one `.compute()`/`.load()` call on `progress`,
    as a stand-in for `dask.diagnostics.ProgressBar()`'s own separate text bar -
    letting a dask load's progress live in the same `Progress` instance (and
    terminal region) as the sweep-wide task-configuration bar from
    `sweep_progress()`.
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
def dask_load_progress(progress: Progress, description: str) -> Iterator[None]:
    """Track one dask-backed `.compute()`/`.load()` call's tasks on `progress`.

    Adds a temporary task to `progress` for the duration of the `with` block,
    advanced once per completed dask task, then removes it - the dask
    equivalent of `dask.diagnostics.ProgressBar()`, but rendered as another
    task on the same shared `rich.progress.Progress` sweep bar instead of its
    own separate text bar.

    Parameters
    ----------
    progress : Progress
        The shared `Progress` instance (typically from `sweep_progress()`) to
        add a task to.
    description : str
        Description shown next to this call's task-completion bar.

    Yields
    ------
    None
    """
    task = progress.add_task(description, total=None)
    try:
        with _RichDaskCallback(progress, task):
            yield
    finally:
        progress.update(task, visible=False)
        progress.remove_task(task)


class TaskCounter(Callback):
    """Dask scheduler callback counting completed tasks as a plain attribute.

    Unlike `_RichDaskCallback`, this touches no UI - it exists so a thread
    other than the one running `.load()`/`.compute()` (e.g. one enforcing a
    wall-clock budget on a background thread) can read `.completed` at any
    moment to infer how much of the computation finished, without reaching
    into `rich.progress.Progress`'s own internal task state.
    """

    def __init__(self) -> None:
        self.completed = 0
        self._lock = threading.Lock()

    def _posttask(
        self, key: Any, result: Any, dsk: Any, state: dict[str, Any], worker_id: Any
    ) -> None:
        with self._lock:
            self.completed += 1


def run_with_timeout(func: Callable[[], None], max_time: float | None) -> tuple[bool, float]:
    """Run `func` on a daemon thread, joining with a wall-clock timeout.

    `func` keeps running as an abandoned daemon thread if it doesn't finish in
    time - harmless only because the caller is expected to build an inferred
    result and then hard-exit its own process shortly after (see
    `run_case_in_subprocess()`'s docstring for why: dask's local `threads`
    scheduler has no API to actually cancel it).

    Parameters
    ----------
    func : Callable[[], None]
        Zero-argument callable to run, typically wrapping a `dask.config.set(
        ...)` context around a `ds.load()` call.
    max_time : float | None
        Seconds to wait before giving up. None waits indefinitely.

    Returns
    -------
    tuple[bool, float]
        `(timed_out, elapsed_s)` - `timed_out` is True iff `func` was still
        running when `max_time` elapsed.
    """
    start = time.perf_counter()
    thread = threading.Thread(target=func, daemon=True)
    thread.start()
    thread.join(timeout=max_time)
    elapsed = time.perf_counter() - start
    return thread.is_alive(), elapsed


def run_case_in_subprocess(
    target: Callable[..., None], args: tuple[Any, ...], max_time: float | None, label: str
) -> dict[str, Any] | None:
    """Run one sweep case in its own process, enforcing `max_time` as an OS-level kill.

    `target` is called as `target(*args, conn)`, where `conn` is the write end
    of a `multiprocessing.Pipe()` - it is responsible for enforcing `max_time`
    itself internally (typically via `run_with_timeout()`, then `os._exit()`
    on a timeout rather than returning normally). Running the case in a
    genuinely separate OS process, rather than just a thread with a timeout in
    this one, is what makes `max_time` enforceable at all: dask's local
    `threads` scheduler has no API to cancel an in-flight task, so a thread
    that blows past its budget would just keep running in the background,
    corrupting whatever case runs next in the same process. A whole child
    process can instead simply be killed outright, reclaiming its threads and
    sockets atomically. This function's own `max_time + SAFETY_GRACE_S` wait
    is a safety net for a child that hangs *before* ever reaching its own
    timed section (e.g. stuck building the dask graph).

    Parameters
    ----------
    target : Callable[..., None]
        Child-process entry point, called as `target(*args, conn)`.
    args : tuple[Any, ...]
        Positional arguments forwarded to `target`, before `conn`.
    max_time : float | None
        Wall-clock budget in seconds `target` is expected to honor
        internally; used here only to size this function's own safety-net
        timeout. None disables the safety net too (waits indefinitely).
    label : str
        Description used in the safety-net timeout's log message.

    Returns
    -------
    dict[str, Any] | None
        The row `target` sent over its `conn`, or None if the safety net had
        to kill it before it sent anything - callers should build their own
        schema-appropriate fallback row in that case.
    """
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(target=target, args=(*args, child_conn))
    process.start()
    child_conn.close()

    deadline = None if max_time is None else max_time + SAFETY_GRACE_S
    row: dict[str, Any] | None = None
    if parent_conn.poll(deadline):
        row = parent_conn.recv()
    else:
        log.error(f"{label}: exceeded safety-net timeout ({deadline}s), killing")

    process.join(5)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
    parent_conn.close()
    return row


def dedupe_by_processing_baseline(items: list[pystac.Item]) -> list[pystac.Item]:
    """Keep one item per (tile, date), preferring the highest processing baseline.

    Earth Search lists the same tile/date twice when ESA reprocesses the archive
    to a new `s2:processing_baseline`, keeping the superseded item rather than
    removing it (see README.md for the investigation). Selecting by nodata%
    instead of baseline can silently keep the superseded item, and blending
    items across baselines in one composite is a radiometric correctness risk
    (baseline 04.00 changed how DN values encode reflectance), not just wasted
    bandwidth.

    Parameters
    ----------
    items : list[pystac.Item]
        Search results, possibly containing multiple processing baselines per
        (tile, date).

    Returns
    -------
    list[pystac.Item]
        One item per (tile, date), the highest `s2:processing_baseline` of any
        candidates sharing that key.
    """
    best: dict[tuple[str, object], pystac.Item] = {}
    for it in items:
        assert it.datetime is not None
        key = (it.properties.get("grid:code", it.id), it.datetime.date())
        baseline = it.properties.get("s2:processing_baseline", "0")
        if key not in best or baseline > best[key].properties.get("s2:processing_baseline", "0"):
            best[key] = it
    return list(best.values())


def search_wide(client: Client) -> list[pystac.Item]:
    """Search the "wide" scenario's AOI/date and dedupe by processing baseline.

    Parameters
    ----------
    client : Client
        An open `pystac_client.Client` for the STAC API to search.

    Returns
    -------
    list[pystac.Item]
        Deduped items covering the "wide" scenario.
    """
    items = list(
        client.search(collections=[COLLECTION], bbox=WIDE_BBOX, datetime=WIDE_DATETIME).items()
    )
    return dedupe_by_processing_baseline(items)


def search_deep(client: Client) -> list[pystac.Item]:
    """Search the "deep" scenario's tile/date range and dedupe by processing baseline.

    Parameters
    ----------
    client : Client
        An open `pystac_client.Client` for the STAC API to search.

    Returns
    -------
    list[pystac.Item]
        Deduped items covering the "deep" scenario.
    """
    items = list(
        client.search(collections=[COLLECTION], bbox=DEEP_BBOX, datetime=DEEP_DATETIME).items()
    )
    items = [it for it in items if it.properties.get("grid:code") == DEEP_TILE]
    return dedupe_by_processing_baseline(items)


SCENARIO_SEARCH: dict[Scenario, Callable[[Client], list[pystac.Item]]] = {
    "wide": search_wide,
    "deep": search_deep,
}
SCENARIO_CRS: dict[Scenario, str] = {"wide": WIDE_CRS, "deep": DEEP_CRS}
