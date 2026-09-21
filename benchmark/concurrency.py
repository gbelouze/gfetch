"""Sweep dask thread-pool size for a direct remote `odc.stac.load()` (no
local download step). See README.md for context and results.

Each case (one `num_workers` value) runs in its own `multiprocessing` child
process, so `--max-time` can be enforced: dask's local `threads` scheduler has
no API to cancel an in-flight task, so a thread that blows past `max_time`
inside *this* process would just keep running in the background - eating
network/CPU and corrupting every later case's own measurement. A whole child
process can instead simply be killed outright once given up on, reclaiming its
threads/sockets atomically. On a timeout, `mb_s`/`mpix_s` are inferred from the
fraction of dask tasks that had completed by then (tracked via
`common.TaskCounter`) - an estimate, not a real measurement, flagged by the
`timed_out` column.

No dependency on the gfetch package itself - standalone, notebook-style.
"""

# %%
import csv
import logging
import os
from multiprocessing.connection import Connection
from pathlib import Path

import cyclopts
import dask
import dask.system
import odc.stac
import pystac
from common import (
    BANDS,
    EARTH_SEARCH_URL,
    RESOLUTION,
    SAFETY_GRACE_S,
    SCENARIO_CRS,
    SCENARIO_SEARCH,
    Scenario,
    TaskCounter,
    dask_load_progress,
    run_case_in_subprocess,
    run_with_timeout,
    setup_logging,
    sweep_progress,
)
from pystac_client import Client

log = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 2048
DEFAULT_WORKER_COUNTS = [128, 64, 32, dask.system.CPU_COUNT]
RESULTS_DIR = Path(__file__).parent / "results"

app = cyclopts.App()


def _run_case_child(
    items: list[pystac.Item],
    crs: str,
    chunk: int,
    num_workers: int,
    max_time: float | None,
    verbose: bool,
    conn: Connection,
) -> None:
    """Child-process entry point: load `items` once and report one result row.

    Written to run standalone in its own process - own logging, own progress
    bar, own GDAL/S3 config - exactly as if it were the top-level script, since
    a `multiprocessing` "spawn" child inherits the parent's stdio (verified:
    `sys.stdout.isatty()`/`rich`'s terminal detection see the same terminal),
    so nothing here needs to know it isn't the main process.

    Parameters
    ----------
    items : list[pystac.Item]
        Items to load, already searched/deduped by the parent.
    crs : str
        Output CRS to reproject/mosaic into.
    chunk : int
        Dask chunk size (pixels, square).
    num_workers : int
        Dask thread-pool size.
    max_time : float | None
        Wall-clock budget in seconds for the `ds.load()` call. None waits for
        it to finish, however long that takes.
    verbose : bool
        Log at DEBUG instead of INFO. Defaults to False.
    conn : Connection
        Write end of a `multiprocessing.Pipe()`; the one result row is sent
        here before this function returns (or, on timeout, before the process
        force-exits).
    """
    setup_logging(verbose)
    odc.stac.configure_s3_access(aws_unsigned=True)

    ds = odc.stac.load(
        items,
        bands=BANDS,
        crs=crs,
        resolution=RESOLUTION,
        chunks={"x": chunk, "y": chunk},
        groupby="solar_day",
    )
    n_bytes_total = sum(v.nbytes for v in ds.data_vars.values())
    npix_total = ds.sizes["x"] * ds.sizes["y"] * ds.sizes["time"] * len(BANDS)
    n_tasks_total = sum(len(v.data.__dask_graph__()) for v in ds.data_vars.values())
    mb_total = n_bytes_total / 1e6
    log.info(f"num_workers={num_workers}: {n_tasks_total} dask tasks, loading {mb_total:.1f}MB")

    def _load() -> None:
        with dask.config.set(scheduler="threads", num_workers=num_workers):
            ds.load()

    counter = TaskCounter()
    description = f"num_workers={num_workers} load"
    with sweep_progress() as progress, dask_load_progress(progress, description), counter:
        timed_out, elapsed = run_with_timeout(_load, max_time)

    if timed_out:
        completed = counter.completed
        frac = completed / n_tasks_total if n_tasks_total else 0.0
        log.warning(
            f"num_workers={num_workers}: exceeded max_time={max_time}s "
            f"({completed}/{n_tasks_total} tasks done) - inferring throughput and giving up"
        )
        row = {
            "num_workers": num_workers,
            "n_bytes": round(n_bytes_total * frac),
            "elapsed_s": round(elapsed, 2),
            "mpix_s": round(npix_total * frac / elapsed / 1e6, 4),
            "mb_s": round(n_bytes_total * frac / elapsed / 1e6, 2),
            "timed_out": True,
            "completed_tasks": completed,
            "total_tasks": n_tasks_total,
        }
        conn.send(row)
        conn.close()
        # dask's own internal worker-thread pool has no cancellation API and may
        # still be running - a normal return would block at interpreter exit
        # waiting to join it (concurrent.futures/dask both register atexit
        # shutdown hooks); os._exit() skips that entirely.
        os._exit(1)

    row = {
        "num_workers": num_workers,
        "n_bytes": n_bytes_total,
        "elapsed_s": round(elapsed, 2),
        "mpix_s": round(npix_total / elapsed / 1e6, 4),
        "mb_s": round(n_bytes_total / elapsed / 1e6, 2),
        "timed_out": False,
        "completed_tasks": n_tasks_total,
        "total_tasks": n_tasks_total,
    }
    log.info(row)
    conn.send(row)
    conn.close()


def run_case(
    items: list[pystac.Item],
    crs: str,
    chunk: int,
    num_workers: int,
    max_time: float | None,
    verbose: bool,
) -> dict:
    """Run one `num_workers` case in its own process and collect the result row.

    Parameters
    ----------
    items : list[pystac.Item]
        Items to load, already searched/deduped by the caller.
    crs : str
        Output CRS to reproject/mosaic into.
    chunk : int
        Dask chunk size (pixels, square).
    num_workers : int
        Dask thread-pool size.
    max_time : float | None
        Wall-clock budget in seconds forwarded to the child; see
        `_run_case_child`.
    verbose : bool
        Log at DEBUG instead of INFO in the child. Defaults to False.

    Returns
    -------
    dict
        One CSV row of results.
    """
    row = run_case_in_subprocess(
        _run_case_child,
        (items, crs, chunk, num_workers, max_time, verbose),
        max_time,
        label=f"num_workers={num_workers}",
    )
    if row is None:
        row = {
            "num_workers": num_workers,
            "n_bytes": 0,
            "elapsed_s": None if max_time is None else max_time + SAFETY_GRACE_S,
            "mpix_s": 0.0,
            "mb_s": 0.0,
            "timed_out": True,
            "completed_tasks": 0,
            "total_tasks": 0,
        }
    return row


@app.default
def main(
    scenario: Scenario = "wide",
    worker_counts: list[int] = DEFAULT_WORKER_COUNTS,
    chunk: int = DEFAULT_CHUNK_SIZE,
    max_time: float | None = None,
    verbose: bool = False,
) -> None:
    """Sweep dask thread count, writing results to `results/concurrency_<scenario>.csv`.

    Parameters
    ----------
    scenario : Scenario
        Which scenario (see `common.py`) to load. Defaults to `"wide"`.
    worker_counts : list[int]
        Dask thread-pool sizes to sweep. Defaults to `DEFAULT_WORKER_COUNTS`.
    chunk : int
        Fixed chunk size (pixels, square) held constant across the sweep.
        Defaults to `DEFAULT_CHUNK_SIZE`.
    max_time : float | None
        Wall-clock budget in seconds for a single case's `ds.load()` call. On
        a timeout, that case's row reports `mb_s`/`mpix_s` inferred from the
        fraction of dask tasks completed by then, flagged via its `timed_out`
        column - not a real measurement. None (default) waits for every case
        to finish, however long that takes.
    verbose : bool
        Log at DEBUG instead of INFO. Defaults to False.
    """
    setup_logging(verbose)
    log.info(f"searching '{EARTH_SEARCH_URL}' for the '{scenario}' scenario")
    client = Client.open(EARTH_SEARCH_URL)
    items = SCENARIO_SEARCH[scenario](client)
    crs = SCENARIO_CRS[scenario]
    log.info(f"found {len(items)} deduped items")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_csv = RESULTS_DIR / f"concurrency_{scenario}.csv"
    fieldnames = [
        "num_workers",
        "n_bytes",
        "elapsed_s",
        "mpix_s",
        "mb_s",
        "timed_out",
        "completed_tasks",
        "total_tasks",
    ]
    with results_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, num_workers in enumerate(worker_counts, start=1):
            log.info(
                f"[{i}/{len(worker_counts)}] num_workers={num_workers}: starting "
                f"(chunk={chunk}, max_time={max_time})"
            )
            row = run_case(items, crs, chunk, num_workers, max_time, verbose)
            writer.writerow(row)
            f.flush()
            log.info(f"[{i}/{len(worker_counts)}] num_workers={num_workers}: {row}")

    log.info(f"results written to {results_csv}")


# %%
if __name__ == "__main__":
    app()
