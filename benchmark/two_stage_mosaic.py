"""Benchmark gfetch's actual two-stage (download-then-load) design against
the "wide" and "deep" scenarios (see common.py), sourced from Earth Search.

Stage 1 (download): stac_asset.download_item(), one call per item, run
concurrently via an asyncio semaphore - full-file sustained downloads
straight to a local cache, matching gfetch's actual download-stage library
choice (see claude/tech-stack.md), instead of odc.stac.load()'s windowed
remote range-reads. stac_asset uses its own HTTP client (aiohttp), not
GDAL/rasterio at all, so GDAL_HTTP_*/VSI_CACHE tuning (gdal_config.py)
has no effect here - the concurrency knob is the asyncio semaphore instead.

Stage 2 (load): odc.stac.load() on the local, href-rewritten items - no
network I/O left, so GDAL_HTTP_* tuning stops applying entirely; the only
remaining knobs are dask chunk size / worker count for the now
CPU+disk-bound reproject/composite step (and possibly GDAL_NUM_THREADS for
GDAL's own internal warp threading). Each load case (one chunk x worker-count
combination) runs in its own `multiprocessing` child process, so `--max-time`
can be enforced - see `concurrency.py`'s module docstring for why dask's local
`threads` scheduler forces this. The download stage above isn't given the
same treatment: it's asyncio/aiohttp-based, not a dask compute, so a plain
`asyncio.wait_for()`-style timeout would actually cancel cleanly there -
different problem, not addressed here.

No dependency on the gfetch package itself - standalone, notebook-style.
"""

# %%
import asyncio
import copy
import csv
import logging
import os
import shutil
import tempfile
import time
from multiprocessing.connection import Connection
from pathlib import Path

import cyclopts
import dask
import dask.system
import odc.stac
import pystac
import stac_asset
from common import (
    BANDS,
    EARTH_SEARCH_URL,
    SAFETY_GRACE_S,
    SCENARIO_CRS,
    SCENARIO_SEARCH,
    Scenario,
    TaskCounter,
    run_case_in_subprocess,
    run_with_timeout,
    setup_logging,
    sweep_progress,
)
from pystac_client import Client
from rich.progress import Progress, TaskID
from stac_asset import Config

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).parent / "cache"
RESULTS_DIR = Path(__file__).parent / "results"

DEFAULT_SCENARIOS: list[Scenario] = ["wide", "deep"]
DEFAULT_DOWNLOAD_CONCURRENCIES = [2, 8, 32]
DEFAULT_BEST_DOWNLOAD_CONCURRENCY = 32
DEFAULT_LOAD_CHUNK_SIZES = [512, 2048, 7168]
DEFAULT_LOAD_WORKER_COUNTS = [dask.system.CPU_COUNT, 32]

app = cyclopts.App()


async def _download_one(
    item: pystac.Item, cache_dir: Path, bands: list[str], sem: asyncio.Semaphore
) -> pystac.Item:
    item_dir = cache_dir / item.id
    item_dir.mkdir(parents=True, exist_ok=True)
    pending = [k for k in bands if not (item_dir / f"{k}.complete").exists()]

    async with sem:
        if pending:
            log.debug(f"{item.id}: downloading {pending}")
            with tempfile.TemporaryDirectory(dir=item_dir) as tmp:
                result = await stac_asset.download_item(
                    item,
                    Path(tmp),
                    config=Config(include=pending),
                    keep_non_downloaded=True,
                )
                for key in pending:
                    href = result.assets[key].get_absolute_href()
                    assert href is not None
                    tmp_path = Path(href)

                    final_path = item_dir / f"{key}{tmp_path.suffix}"
                    shutil.move(str(tmp_path), str(final_path))
                    (item_dir / f"{key}.complete").touch()
                    result.assets[key].href = str(final_path)
        else:
            log.debug(f"{item.id}: already cached, skipping")
            result = item

    for key in bands:
        if key not in pending:
            # f"{key}.*" also matches the f"{key}.complete" sentinel touched above -
            # exclude it explicitly rather than relying on there being exactly one match.
            (path,) = (p for p in item_dir.glob(f"{key}.*") if p.suffix != ".complete")
            result.assets[key].href = str(path)
    result.set_self_href(str(item_dir / f"{item.id}.json"))
    return result


async def download_items(
    items: list[pystac.Item], cache_dir: Path, bands: list[str], max_concurrency: int
) -> list[pystac.Item]:
    """Download `items`' `bands` assets to `cache_dir`, `max_concurrency` at a time.

    Parameters
    ----------
    items : list[pystac.Item]
        Items to download. Deep-copied internally before download, since
        `stac_asset.download_item()` mutates its input in place (rewrites
        asset hrefs to the local download path) - callers can safely reuse
        the same search-result items across repeated/swept download attempts
        into different cache dirs.
    cache_dir : Path
        Directory to download into, one subdirectory per item id.
    bands : list[str]
        Asset keys to download for each item.
    max_concurrency : int
        Maximum number of items downloaded concurrently.

    Returns
    -------
    list[pystac.Item]
        The downloaded items, with asset hrefs rewritten to local paths.
    """
    sem = asyncio.Semaphore(max_concurrency)
    return await asyncio.gather(
        *(_download_one(copy.deepcopy(it), cache_dir, bands, sem) for it in items)
    )


def cache_size_bytes(cache_dir: Path) -> int:
    """Total size in bytes of every file under `cache_dir`.

    Parameters
    ----------
    cache_dir : Path
        Directory to sum file sizes under, recursively.

    Returns
    -------
    int
        Total size in bytes.
    """
    return sum(p.stat().st_size for p in cache_dir.rglob("*") if p.is_file())


def _run_load_case_child(
    items: list[pystac.Item],
    crs: str,
    chunk: int,
    num_workers: int,
    max_time: float | None,
    verbose: bool,
    conn: Connection,
) -> None:
    """Child-process entry point: load already-downloaded `items` once and report one result row.

    Parameters
    ----------
    items : list[pystac.Item]
        Items whose asset hrefs already point at local files.
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
    # No S3 access needed here (items are already local), but GDAL config doesn't
    # cross the process boundary into this child - without this, GDAL_DISABLE_READDIR_ON_OPEN
    # and the rest of cloud_defaults (see configure_s3_access()'s own default) never
    # apply here, unlike the pre-multiprocessing in-process version of this stage.
    odc.stac.configure_rio(cloud_defaults=True)
    ds = odc.stac.load(
        items,
        bands=BANDS,
        crs=crs,
        resolution=10,
        chunks={"x": chunk, "y": chunk},
        groupby="solar_day",
    )
    n_bytes_total = sum(v.nbytes for v in ds.data_vars.values())
    npix_total = ds.sizes["x"] * ds.sizes["y"] * ds.sizes["time"] * len(BANDS)
    n_tasks_total = sum(len(v.data.__dask_graph__()) for v in ds.data_vars.values())

    def _load() -> None:
        with dask.config.set(scheduler="threads", num_workers=num_workers):
            ds.load()

    # No rich progress bar here (unlike concurrency.py/chunks.py/gdal_config.py's
    # own case children): the parent keeps a single `sweep_progress()` Live display
    # open across all three stages of this script's run, and a second `Live` region
    # from this child would fight it for the terminal. TaskCounter alone still gives
    # us what a timeout needs to know.
    counter = TaskCounter()
    with counter:
        timed_out, elapsed = run_with_timeout(_load, max_time)

    if timed_out:
        completed = counter.completed
        frac = completed / n_tasks_total if n_tasks_total else 0.0
        log.warning(
            f"chunk={chunk} num_workers={num_workers}: exceeded max_time={max_time}s "
            f"({completed}/{n_tasks_total} tasks done) - inferring throughput and giving up"
        )
        row = {
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


def run_load_case(
    items: list[pystac.Item],
    crs: str,
    chunk: int,
    num_workers: int,
    max_time: float | None,
    verbose: bool,
) -> dict:
    """Run one (chunk, num_workers) load case in its own process and collect the result row.

    Parameters
    ----------
    items : list[pystac.Item]
        Items whose asset hrefs already point at local files.
    crs : str
        Output CRS to reproject/mosaic into.
    chunk : int
        Dask chunk size (pixels, square).
    num_workers : int
        Dask thread-pool size.
    max_time : float | None
        Wall-clock budget in seconds forwarded to the child; see
        `_run_load_case_child`.
    verbose : bool
        Log at DEBUG instead of INFO in the child. Defaults to False.

    Returns
    -------
    dict
        `{"n_bytes", "elapsed_s", "mpix_s", "mb_s", "timed_out",
        "completed_tasks", "total_tasks"}`.
    """
    row = run_case_in_subprocess(
        _run_load_case_child,
        (items, crs, chunk, num_workers, max_time, verbose),
        max_time,
        label=f"chunk={chunk} num_workers={num_workers}",
    )
    if row is None:
        row = {
            "n_bytes": 0,
            "elapsed_s": None if max_time is None else max_time + SAFETY_GRACE_S,
            "mpix_s": 0.0,
            "mb_s": 0.0,
            "timed_out": True,
            "completed_tasks": 0,
            "total_tasks": 0,
        }
    return row


def run_download_sweep(
    scenario_items: dict[Scenario, list[pystac.Item]],
    cache_dir: Path,
    download_concurrencies: list[int],
    progress: Progress,
    task: TaskID,
) -> None:
    """Sweep download concurrency on a small item subset per scenario.

    Parameters
    ----------
    scenario_items : dict[Scenario, list[pystac.Item]]
        Deduped search-result items per scenario name.
    cache_dir : Path
        Base directory to download disposable sweep runs into (removed after
        each run).
    download_concurrencies : list[int]
        Values of `max_concurrency` to sweep.
    progress : Progress
        Sweep-wide progress bar to advance after every run.
    task : TaskID
        Task ID on `progress` to advance after every run.
    """
    download_sweep_csv = RESULTS_DIR / "two_stage_download_concurrency.csv"
    with download_sweep_csv.open("w", newline="") as f:
        fieldnames = ["scenario", "max_concurrency", "n_bytes", "elapsed_s", "mb_s"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for name, its in scenario_items.items():
            subset = its[:3]
            for max_concurrency in download_concurrencies:
                log.info(f"[{name}] download sweep: max_concurrency={max_concurrency}")
                sweep_dir = cache_dir / f"_sweep_{name}_{max_concurrency}"
                shutil.rmtree(sweep_dir, ignore_errors=True)

                start = time.perf_counter()
                asyncio.run(download_items(subset, sweep_dir, BANDS, max_concurrency))
                elapsed = time.perf_counter() - start

                n_bytes = cache_size_bytes(sweep_dir)
                row = {
                    "scenario": name,
                    "max_concurrency": max_concurrency,
                    "n_bytes": n_bytes,
                    "elapsed_s": round(elapsed, 2),
                    "mb_s": round(n_bytes / elapsed / 1e6, 2),
                }
                writer.writerow(row)
                f.flush()
                log.info(row)
                shutil.rmtree(sweep_dir, ignore_errors=True)
                progress.advance(task)

    log.info(f"download concurrency sweep written to {download_sweep_csv}")


def run_full_download(
    scenario_items: dict[Scenario, list[pystac.Item]],
    cache_dir: Path,
    best_download_concurrency: int,
    progress: Progress,
    task: TaskID,
) -> dict[Scenario, list[pystac.Item]]:
    """Download each scenario's full deduped item set once, for stage 2 to reuse.

    Parameters
    ----------
    scenario_items : dict[Scenario, list[pystac.Item]]
        Deduped search-result items per scenario name.
    cache_dir : Path
        Base directory to download the full, reusable cache into (one
        subdirectory per scenario name).
    best_download_concurrency : int
        `max_concurrency` picked from the download sweep.
    progress : Progress
        Sweep-wide progress bar to advance after every scenario.
    task : TaskID
        Task ID on `progress` to advance after every scenario.

    Returns
    -------
    dict[Scenario, list[pystac.Item]]
        Downloaded items per scenario name, with asset hrefs rewritten to
        local paths.
    """
    download_full_csv = RESULTS_DIR / "two_stage_download_full.csv"
    with download_full_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["scenario", "n_items", "n_bytes", "elapsed_s", "mb_s"]
        )
        writer.writeheader()

        local_items: dict[Scenario, list[pystac.Item]] = {}
        for name, its in scenario_items.items():
            log.info(
                f"[{name}] downloading {len(its)} items at concurrency={best_download_concurrency}"
            )
            scenario_cache_dir = cache_dir / name
            start = time.perf_counter()
            local_items[name] = asyncio.run(
                download_items(its, scenario_cache_dir, BANDS, best_download_concurrency)
            )
            elapsed = time.perf_counter() - start

            n_bytes = cache_size_bytes(scenario_cache_dir)
            row = {
                "scenario": name,
                "n_items": len(its),
                "n_bytes": n_bytes,
                "elapsed_s": round(elapsed, 2),
                "mb_s": round(n_bytes / elapsed / 1e6, 2),
            }
            writer.writerow(row)
            f.flush()
            log.info(row)
            progress.advance(task)

    log.info(f"full download written to {download_full_csv}")
    return local_items


def run_load_sweep(
    local_items: dict[Scenario, list[pystac.Item]],
    load_chunk_sizes: list[int],
    load_worker_counts: list[int],
    max_time: float | None,
    verbose: bool,
    progress: Progress,
    task: TaskID,
) -> None:
    """Sweep chunk size / worker count for the local, purely-CPU+disk load stage.

    Parameters
    ----------
    local_items : dict[Scenario, list[pystac.Item]]
        Downloaded items per scenario name, with asset hrefs already local.
    load_chunk_sizes : list[int]
        Dask chunk sizes (pixels, square) to sweep.
    load_worker_counts : list[int]
        Dask thread-pool sizes to sweep.
    max_time : float | None
        Wall-clock budget in seconds for a single case's `ds.load()` call,
        forwarded to `run_load_case`.
    verbose : bool
        Log at DEBUG instead of INFO in each case's child process.
    progress : Progress
        Sweep-wide progress bar to advance after every run.
    task : TaskID
        Task ID on `progress` to advance after every run.
    """
    load_csv = RESULTS_DIR / "two_stage_load.csv"
    with load_csv.open("w", newline="") as f:
        fieldnames = [
            "scenario",
            "chunk",
            "num_workers",
            "n_bytes",
            "elapsed_s",
            "mpix_s",
            "mb_s",
            "timed_out",
            "completed_tasks",
            "total_tasks",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for name, its in local_items.items():
            crs = SCENARIO_CRS[name]
            for chunk in load_chunk_sizes:
                for num_workers in load_worker_counts:
                    log.info(
                        f"[{name}] load sweep: chunk={chunk} num_workers={num_workers}: starting "
                        f"(max_time={max_time})"
                    )
                    result = run_load_case(its, crs, chunk, num_workers, max_time, verbose)

                    row = {"scenario": name, "chunk": chunk, "num_workers": num_workers, **result}
                    writer.writerow(row)
                    f.flush()
                    log.info(f"[{name}] load sweep: chunk={chunk} num_workers={num_workers}: {row}")
                    progress.advance(task)

    log.info(f"load sweep written to {load_csv}")


@app.default
def main(
    scenarios: list[Scenario] = DEFAULT_SCENARIOS,
    download_concurrencies: list[int] = DEFAULT_DOWNLOAD_CONCURRENCIES,
    best_download_concurrency: int = DEFAULT_BEST_DOWNLOAD_CONCURRENCY,
    load_chunk_sizes: list[int] = DEFAULT_LOAD_CHUNK_SIZES,
    load_worker_counts: list[int] = DEFAULT_LOAD_WORKER_COUNTS,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_time: float | None = None,
    verbose: bool = False,
) -> None:
    """Benchmark the two-stage (download-then-load) design across scenarios.

    Runs, in order: a download-concurrency sweep on a small item subset, a
    full download of each scenario's deduped item set at
    `best_download_concurrency`, then a chunk-size/worker-count sweep of the
    local load stage.

    Parameters
    ----------
    scenarios : list[Scenario]
        Scenarios to benchmark (see `common.py`). Defaults to
        `DEFAULT_SCENARIOS` (both).
    download_concurrencies : list[int]
        `max_concurrency` values to sweep in the download-concurrency sweep.
        Defaults to `DEFAULT_DOWNLOAD_CONCURRENCIES`.
    best_download_concurrency : int
        `max_concurrency` to use for the full download that stage 2 reuses.
        Defaults to `DEFAULT_BEST_DOWNLOAD_CONCURRENCY`.
    load_chunk_sizes : list[int]
        Dask chunk sizes (pixels, square) to sweep in the load stage.
        Defaults to `DEFAULT_LOAD_CHUNK_SIZES`.
    load_worker_counts : list[int]
        Dask thread-pool sizes to sweep in the load stage. Defaults to
        `DEFAULT_LOAD_WORKER_COUNTS`.
    cache_dir : Path
        Directory to download into. Defaults to `DEFAULT_CACHE_DIR`.
    max_time : float | None
        Wall-clock budget in seconds for a single load-stage case's
        `ds.load()` call (not the download stage - see module docstring). On
        a timeout, that case's row reports `mb_s`/`mpix_s` inferred from the
        fraction of dask tasks completed by then, flagged via its `timed_out`
        column - not a real measurement. None (default) waits for every case
        to finish, however long that takes.
    verbose : bool
        Log at DEBUG instead of INFO. Defaults to False.
    """
    setup_logging(verbose)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    odc.stac.configure_s3_access(aws_unsigned=True)

    log.info(f"searching '{EARTH_SEARCH_URL}' for scenarios: {scenarios}")
    client = Client.open(EARTH_SEARCH_URL)
    scenario_items = {name: SCENARIO_SEARCH[name](client) for name in scenarios}
    for name, its in scenario_items.items():
        log.info(f"[{name}] {len(its)} deduped items")

    n_download_sweep_runs = len(scenarios) * len(download_concurrencies)
    n_load_sweep_runs = len(scenarios) * len(load_chunk_sizes) * len(load_worker_counts)
    log.info(
        f"plan: {n_download_sweep_runs} download-sweep runs, {len(scenarios)} full downloads, "
        f"{n_load_sweep_runs} load-sweep runs"
    )

    with sweep_progress() as progress:
        download_sweep_task = progress.add_task(
            "download concurrency sweep", total=n_download_sweep_runs
        )
        full_download_task = progress.add_task("full download", total=len(scenarios))
        load_sweep_task = progress.add_task("load sweep", total=n_load_sweep_runs)

        run_download_sweep(
            scenario_items, cache_dir, download_concurrencies, progress, download_sweep_task
        )
        local_items = run_full_download(
            scenario_items, cache_dir, best_download_concurrency, progress, full_download_task
        )
        run_load_sweep(
            local_items,
            load_chunk_sizes,
            load_worker_counts,
            max_time,
            verbose,
            progress,
            load_sweep_task,
        )

    log.info("two-stage benchmark complete")


# %%
if __name__ == "__main__":
    app()
