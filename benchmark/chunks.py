"""Sweep dask chunk size for a direct remote `odc.stac.load()`. No `bbox=`
passed to `load()`, so the geobox is the union of items' native footprints
(unclipped) - for the "wide" scenario (the default), this reproduces the
original benchmark's reported 10980x90978px/5.58GiB shape. See README.md for
context and results.

Each case (one `chunk` value) runs in its own `multiprocessing` child process,
so `--max-time` can be enforced - see `concurrency.py`'s module docstring for
why dask's local `threads` scheduler forces this. On a timeout, `mb_s`/
`mpix_s` are inferred from the fraction of dask tasks that had completed by
then, flagged by the `timed_out` column.

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

DEFAULT_CHUNK_SIZES = [512, 1024, 2048, 3072, 4096, 5120, 6144, 7168]
DEFAULT_WORKERS = 64
RESULTS_DIR = Path(__file__).parent / "results"

app = cyclopts.App()


def _run_case_child(
    items: list[pystac.Item],
    crs: str,
    chunk: int,
    workers: int,
    max_time: float | None,
    verbose: bool,
    conn: Connection,
) -> None:
    """Child-process entry point: load `items` at one chunk size and report one result row.

    Parameters
    ----------
    items : list[pystac.Item]
        Items to load, already searched/deduped by the parent.
    crs : str
        Output CRS to reproject/mosaic into.
    chunk : int
        Dask chunk size (pixels, square).
    workers : int
        Dask thread-pool size held constant across the sweep.
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

    log.info(f"chunk={chunk}: building dask graph")
    ds = odc.stac.load(
        items,
        bands=BANDS,
        crs=crs,
        resolution=RESOLUTION,
        chunks={"x": chunk, "y": chunk},
        groupby="solar_day",
    )
    n_bytes_total = sum(v.nbytes for v in ds.data_vars.values())
    n_tasks_total = sum(len(v.data.__dask_graph__()) for v in ds.data_vars.values())
    npix_total = ds.sizes["x"] * ds.sizes["y"] * ds.sizes["time"] * len(BANDS)
    mb_total = n_bytes_total / 1e6
    log.info(f"chunk={chunk}: {n_tasks_total} dask tasks, loading {mb_total:.1f}MB")

    def _load() -> None:
        with dask.config.set(scheduler="threads", num_workers=workers):
            ds.load()

    counter = TaskCounter()
    description = f"chunk={chunk} load"
    with sweep_progress() as progress, dask_load_progress(progress, description), counter:
        timed_out, elapsed = run_with_timeout(_load, max_time)

    base_row = {
        "chunk": chunk,
        "num_workers": workers,
        "y": ds.sizes["y"],
        "x": ds.sizes["x"],
        "time_steps": ds.sizes["time"],
    }

    if timed_out:
        completed = counter.completed
        frac = completed / n_tasks_total if n_tasks_total else 0.0
        log.warning(
            f"chunk={chunk}: exceeded max_time={max_time}s "
            f"({completed}/{n_tasks_total} tasks done) - inferring throughput and giving up"
        )
        row = {
            **base_row,
            "n_tasks": n_tasks_total,
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
        **base_row,
        "n_tasks": n_tasks_total,
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
    workers: int,
    max_time: float | None,
    verbose: bool,
) -> dict:
    """Run one `chunk` case in its own process and collect the result row.

    Parameters
    ----------
    items : list[pystac.Item]
        Items to load, already searched/deduped by the caller.
    crs : str
        Output CRS to reproject/mosaic into.
    chunk : int
        Dask chunk size (pixels, square).
    workers : int
        Dask thread-pool size held constant across the sweep.
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
        (items, crs, chunk, workers, max_time, verbose),
        max_time,
        label=f"chunk={chunk}",
    )
    if row is None:
        row = {
            "chunk": chunk,
            "num_workers": workers,
            "n_tasks": 0,
            "y": 0,
            "x": 0,
            "time_steps": 0,
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
    chunks: list[int] = DEFAULT_CHUNK_SIZES,
    workers: int = DEFAULT_WORKERS,
    max_time: float | None = None,
    verbose: bool = False,
) -> None:
    """Sweep chunk sizes and write throughput results to `results/chunks_<scenario>.csv`.

    Parameters
    ----------
    scenario : Scenario
        Which scenario (see `common.py`) to load. Defaults to `"wide"`.
    chunks : list[int]
        Chunk sizes (pixels, square) to sweep. Defaults to
        `DEFAULT_CHUNK_SIZES`.
    workers : int
        Dask thread-pool size held constant across the sweep. Without an
        explicit value, dask's threaded scheduler falls back to
        `dask.system.CPU_COUNT` threads, which `concurrency.py`'s own
        sweep found to be the slowest concurrency it tested - left implicit
        here, a smaller `chunks` value (more, smaller GDAL range-requests)
        would compound with that low concurrency rather than isolating chunk
        size as the only variable. Defaults to `DEFAULT_WORKERS`.
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
    results_csv = RESULTS_DIR / f"chunks_{scenario}.csv"
    fieldnames = [
        "chunk",
        "num_workers",
        "n_tasks",
        "y",
        "x",
        "time_steps",
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

        for i, chunk in enumerate(chunks, start=1):
            log.info(
                f"[{i}/{len(chunks)}] chunk={chunk}: starting (workers={workers}, "
                f"max_time={max_time})"
            )
            row = run_case(items, crs, chunk, workers, max_time, verbose)
            writer.writerow(row)
            f.flush()
            log.info(f"[{i}/{len(chunks)}] chunk={chunk}: {row}")

    log.info(f"results written to {results_csv}")


# %%
if __name__ == "__main__":
    app()
