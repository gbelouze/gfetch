"""Compare GDAL configs for a direct remote `odc.stac.load()`, on a small
item subset for turnaround speed: baseline settings, HTTP/2 multiplexing in
isolation, flytemosaic's full config
(github.com/ljstrnadiii/flytemosaic, `gdal_configs.py`), and flytemosaic's
production concurrency pattern - a single-threaded dask scheduler relying on
`GDAL_NUM_THREADS=ALL_CPUS` instead of Python-thread fan-out (see
`flyte/build.py::write_mosaic_partition_task` in that repo). See README.md
for the full rationale and results.

Each case (one config x worker-count combination) runs in its own
`multiprocessing` child process, so `--max-time` can be enforced - see
`concurrency.py`'s module docstring for why dask's local `threads` scheduler
forces this. On a timeout, `mb_s`/`mpix_s` are inferred from the fraction of
dask tasks that had completed by then, flagged by the `timed_out` column.

No dependency on the gfetch package itself - standalone, notebook-style.
"""

# %%
import csv
import logging
import os
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import cyclopts
import dask
import dask.system
import odc.stac
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

DEFAULT_N_ITEMS = 6
DEFAULT_WORKER_COUNTS = [dask.system.CPU_COUNT, 64]
DEFAULT_MEMORY_GB = 4
RESULTS_DIR = Path(__file__).parent / "results"

app = cyclopts.App()


def aggressive_gdal_config(memory_gb: int, debug: bool = False) -> dict[str, str | int]:
    """Build flytemosaic's `get_worker_config()`, unchanged from the source.

    Parameters
    ----------
    memory_gb : int
        Memory budget in gigabytes, used to size the GDAL/VSI cache options.
    debug : bool
        Whether to enable GDAL's verbose HTTP/CPL debug logging. Defaults to
        False.

    Returns
    -------
    dict[str, str | int]
        GDAL/CPL config options to pass to `odc.stac.configure_rio()`.
    """
    return {
        "GDAL_HTTP_MAX_RETRY": "20",
        "GDAL_HTTP_RETRY_DELAY": "30",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_HTTP_MULTIPLEX": "YES",
        "GDAL_HTTP_VERSION": "2",
        "GDAL_DISABLE_READDIR_ON_OPEN": "TRUE",
        "CPL_VSIL_CURL_CACHE_SIZE": str(1024**3 * memory_gb // 3),
        "CPL_VSIL_CURL_CHUNK_SIZE": str(1024**2 * 12),
        # no-op in this pipeline: odc-loader forces VSI_CACHE=False on every
        # pixel read regardless of global config, kept only for parity with
        # the source.
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": str(1024**3 * memory_gb // 3),
        # rasterio special-cases this key, calling GDALSetCacheMax64() directly with
        # the raw value instead of str(val).encode() like every other option here -
        # passing a str (as flytemosaic's original does, for plain env-var export)
        # raises "TypeError: an integer is required" through configure_rio()/rasterio.Env().
        "GDAL_CACHEMAX": 1024**3 * memory_gb // 2,
        "GDAL_NUM_THREADS": "ALL_CPUS",
        "CPL_DEBUG": "ON" if debug else "OFF",
        "CPL_CURL_VERBOSE": "YES" if debug else "NO",
    }


def multiplex_only_gdal_config() -> dict[str, str | int]:
    """Isolate just the HTTP/2-multiplexing settings from `aggressive_gdal_config()`.

    Returns
    -------
    dict[str, str | int]
        GDAL/CPL config options to pass to `odc.stac.configure_rio()`.
    """
    return {
        "GDAL_HTTP_MULTIPLEX": "YES",
        "GDAL_HTTP_VERSION": "2",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    }


def build_cases(memory_gb: int, debug: bool) -> list[tuple[str, dict[str, str | int], str | None]]:
    """Build the (label, rio config kwargs, scheduler) cases to sweep.

    Parameters
    ----------
    memory_gb : int
        Memory budget forwarded to `aggressive_gdal_config()`.
    debug : bool
        Verbose GDAL/CPL debug logging, forwarded to `aggressive_gdal_config()`.

    Returns
    -------
    list[tuple[str, dict[str, str | int], str | None]]
        One entry per case. The scheduler is None to sweep `worker_counts`
        with `scheduler="threads"`, or `"single-threaded"` to run once with no
        thread fan-out.
    """
    cases: list[tuple[str, dict[str, str | int], str | None]] = [
        ("baseline", {}, None),
        ("multiplex_only", multiplex_only_gdal_config(), None),
        ("aggressive", aggressive_gdal_config(memory_gb=memory_gb, debug=debug), None),
        (
            "single_threaded_gdal_threads",
            aggressive_gdal_config(memory_gb=memory_gb, debug=debug),
            "single-threaded",
        ),
    ]
    return cases


def _run_case_child(
    items: list,
    crs: str,
    label: str,
    rio_kwargs: dict[str, str | int],
    scheduler: str,
    num_workers: int | None,
    max_time: float | None,
    verbose: bool,
    conn: Connection,
) -> None:
    """Child-process entry point: load `items` under one GDAL config and report one result row.

    Parameters
    ----------
    items : list
        STAC items to load, already filtered to the benchmark's small subset.
    crs : str
        Output CRS to reproject/mosaic into.
    label : str
        Config label to record in the result row.
    rio_kwargs : dict[str, str | int]
        GDAL/CPL config options for this case, passed to
        `odc.stac.configure_rio()` - set here, inside the child, since GDAL
        config doesn't cross process boundaries.
    scheduler : str
        Dask scheduler name, e.g. "threads" or "single-threaded".
    num_workers : int | None
        Thread count for the "threads" scheduler, or None for schedulers that
        don't take one (e.g. "single-threaded").
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
    # configure_rio() defaults cloud_defaults to False (unlike
    # configure_s3_access()); pass it explicitly so "baseline" still gets
    # GDAL_DISABLE_READDIR_ON_OPEN and stays a fair floor for the other cases.
    odc.stac.configure_rio(
        aws={"aws_unsigned": True}, cloud_defaults=True, verbose=False, **rio_kwargs
    )

    log.info(f"config={label} scheduler={scheduler} num_workers={num_workers}: running")
    ds = odc.stac.load(
        items,
        bands=BANDS,
        crs=crs,
        resolution=RESOLUTION,
        chunks={"x": 2048, "y": 2048},
        groupby="solar_day",
    )
    n_bytes_total = sum(v.nbytes for v in ds.data_vars.values())
    npix_total = ds.sizes["x"] * ds.sizes["y"] * ds.sizes["time"] * len(BANDS)
    n_tasks_total = sum(len(v.data.__dask_graph__()) for v in ds.data_vars.values())

    scheduler_kwargs: dict[str, Any] = {"scheduler": scheduler}
    if num_workers is not None:
        scheduler_kwargs["num_workers"] = num_workers

    def _load() -> None:
        with dask.config.set(**scheduler_kwargs):
            ds.load()

    counter = TaskCounter()
    description = f"config={label} num_workers={num_workers} load"
    with sweep_progress() as progress, dask_load_progress(progress, description), counter:
        timed_out, elapsed = run_with_timeout(_load, max_time)

    row_num_workers = num_workers if num_workers is not None else 1

    if timed_out:
        completed = counter.completed
        frac = completed / n_tasks_total if n_tasks_total else 0.0
        log.warning(
            f"config={label} num_workers={num_workers}: exceeded max_time={max_time}s "
            f"({completed}/{n_tasks_total} tasks done) - inferring throughput and giving up"
        )
        row = {
            "config": label,
            "num_workers": row_num_workers,
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
        "config": label,
        "num_workers": row_num_workers,
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
    items: list,
    crs: str,
    label: str,
    rio_kwargs: dict[str, str | int],
    scheduler: str,
    num_workers: int | None,
    max_time: float | None,
    verbose: bool,
) -> dict:
    """Run one (config, num_workers) case in its own process and collect the result row.

    Parameters
    ----------
    items : list
        STAC items to load, already filtered to the benchmark's small subset.
    crs : str
        Output CRS to reproject/mosaic into.
    label : str
        Config label to record in the result row.
    rio_kwargs : dict[str, str | int]
        GDAL/CPL config options for this case, forwarded to the child.
    scheduler : str
        Dask scheduler name, e.g. "threads" or "single-threaded".
    num_workers : int | None
        Thread count for the "threads" scheduler, or None for schedulers that
        don't take one (e.g. "single-threaded").
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
        (items, crs, label, rio_kwargs, scheduler, num_workers, max_time, verbose),
        max_time,
        label=f"config={label} num_workers={num_workers}",
    )
    if row is None:
        row = {
            "config": label,
            "num_workers": num_workers if num_workers is not None else 1,
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
    n_items: int = DEFAULT_N_ITEMS,
    worker_counts: list[int] = DEFAULT_WORKER_COUNTS,
    memory_gb: int = DEFAULT_MEMORY_GB,
    debug: bool = False,
    max_time: float | None = None,
    verbose: bool = False,
) -> None:
    """Compare GDAL configs, writing results to `results/gdal_config_<scenario>.csv`.

    Parameters
    ----------
    scenario : Scenario
        Which scenario (see `common.py`) to load. Defaults to `"wide"`.
    n_items : int
        Number of deduped scenario items to use, from the front of the
        search results. Defaults to `DEFAULT_N_ITEMS`.
    worker_counts : list[int]
        Dask thread-pool sizes to sweep for every case except
        `single_threaded_gdal_threads`, which always runs once with no thread
        fan-out. Defaults to `DEFAULT_WORKER_COUNTS`.
    memory_gb : int
        Memory budget in gigabytes forwarded to `aggressive_gdal_config()`.
        Defaults to `DEFAULT_MEMORY_GB`.
    debug : bool
        Enable GDAL's verbose HTTP/CPL debug logging in the `aggressive` and
        `single_threaded_gdal_threads` cases. Defaults to False.
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
    items = SCENARIO_SEARCH[scenario](client)[:n_items]
    crs = SCENARIO_CRS[scenario]
    log.info(f"using {len(items)} deduped items")

    cases = build_cases(memory_gb=memory_gb, debug=debug)
    total_runs = sum(
        1 if scheduler == "single-threaded" else len(worker_counts) for _, _, scheduler in cases
    )
    log.info(f"{len(cases)} configs, {total_runs} total runs (max_time={max_time})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_csv = RESULTS_DIR / f"gdal_config_{scenario}.csv"
    fieldnames = [
        "config",
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

        i = 0
        for label, rio_kwargs, scheduler in cases:
            if scheduler == "single-threaded":
                i += 1
                log.info(f"[{i}/{total_runs}] config={label} num_workers=1: starting")
                row = run_case(
                    items, crs, label, rio_kwargs, "single-threaded", None, max_time, verbose
                )
                writer.writerow(row)
                f.flush()
                log.info(f"[{i}/{total_runs}] config={label} num_workers=1: {row}")
            else:
                for num_workers in worker_counts:
                    i += 1
                    log.info(
                        f"[{i}/{total_runs}] config={label} num_workers={num_workers}: starting"
                    )
                    row = run_case(
                        items, crs, label, rio_kwargs, "threads", num_workers, max_time, verbose
                    )
                    writer.writerow(row)
                    f.flush()
                    log.info(f"[{i}/{total_runs}] config={label} num_workers={num_workers}: {row}")

    log.info(f"results written to {results_csv}")


# %%
if __name__ == "__main__":
    app()
