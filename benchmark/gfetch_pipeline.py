"""Benchmark gfetch's own `search`/`download`/`mosaic` functions end-to-end on the
"wide"/"deep" scenarios also used by this directory's other, notebook-style
benchmarks (concurrency.py/chunks.py/two_stage_mosaic.py) - the goal is to check
whether the actual library, now that odc.stac.load()'s S3-credential fix and
search()'s processing-baseline dedup have been folded back in (see
claude/tech-stack.md's Decision log), reaches throughput comparable to the reference
numbers those other scripts measured, rather than reimplementing the same download/
load logic by hand.

This is the only script in `benchmark/` with a dependency on the `gfetch` package
itself. Unlike concurrency.py/chunks.py/gdal_config.py/two_stage_mosaic.py, cases
here don't run in a timeout-enforcing subprocess - this is a one-shot comparison
check, not an unattended multi-hour sweep, so a hung case is meant to be interrupted
by hand.
"""

# %%
import asyncio
import csv
import logging
import time
from pathlib import Path

import cyclopts
import dask
import dask.system
from common import (
    DEEP_BBOX,
    DEEP_DATETIME,
    DEEP_TILE,
    RESOLUTION,
    WIDE_BBOX,
    WIDE_DATETIME,
    Scenario,
    dask_load_progress,
    setup_logging,
    sweep_progress,
)
from odc.geo.geobox import GeoBox
from stac_asset import Config

from gfetch.download import download_items
from gfetch.mosaic import load
from gfetch.profiles import get_profile
from gfetch.search import search
from gfetch.sources import get_source

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).parent / "cache" / "gfetch_pipeline"
RESULTS_DIR = Path(__file__).parent / "results"

DEFAULT_SCENARIOS: list[Scenario] = ["wide", "deep"]
DEFAULT_DOWNLOAD_CONCURRENCY = 32
DEFAULT_LOAD_CHUNK_SIZES = [512, 2048, 7168]
DEFAULT_LOAD_WORKER_COUNTS = [dask.system.CPU_COUNT, 32]

app = cyclopts.App()


def _search_scenario(scenario: Scenario) -> list:
    """Search gfetch's own `search()` for one scenario's AOI/date range.

    Parameters
    ----------
    scenario : Scenario
        `"wide"` or `"deep"` (see `common.py`).

    Returns
    -------
    list[pystac.Item]
        Matching items - already deduped by processing baseline, since that now
        happens inside `gfetch.search.search()` itself.
    """
    source = get_source("earthsearch")
    if scenario == "wide":
        return search(source, "sentinel-2", bbox=WIDE_BBOX, datetime=WIDE_DATETIME)
    items = search(source, "sentinel-2", bbox=DEEP_BBOX, datetime=DEEP_DATETIME)
    return [it for it in items if it.properties.get("grid:code") == DEEP_TILE]


def _cache_size_bytes(cache_dir: Path) -> int:
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


def run_download(
    scenario_items: dict[Scenario, list],
    cache_dir: Path,
    bands: list[str],
    max_concurrency: int,
) -> dict[Scenario, list]:
    """Download each scenario's items via `gfetch.download.download_items()`.

    Parameters
    ----------
    scenario_items : dict[Scenario, list]
        Search-result items per scenario name.
    cache_dir : Path
        Base directory to download into (one subdirectory per scenario name).
    bands : list[str]
        Asset keys to download.
    max_concurrency : int
        `max_concurrent_items` passed through to `download_items()`.

    Returns
    -------
    dict[Scenario, list]
        Downloaded items per scenario name, with asset hrefs rewritten to local
        paths.
    """
    download_csv = RESULTS_DIR / "gfetch_pipeline_download.csv"
    local_items: dict[Scenario, list] = {}

    with (
        download_csv.open("w", newline="") as f,
        sweep_progress() as progress,
    ):
        writer = csv.DictWriter(
            f, fieldnames=["scenario", "n_items", "n_bytes", "elapsed_s", "mb_s"]
        )
        writer.writeheader()
        task = progress.add_task("gfetch download", total=len(scenario_items))

        for name, items in scenario_items.items():
            scenario_cache_dir = cache_dir / name
            log.info(
                f"[{name}] downloading {len(items)} items at max_concurrency={max_concurrency}"
            )
            start = time.perf_counter()
            local_items[name] = asyncio.run(
                download_items(
                    items,
                    scenario_cache_dir,
                    bands,
                    config=Config(),
                    max_concurrent_items=max_concurrency,
                )
            )
            elapsed = time.perf_counter() - start

            n_bytes = _cache_size_bytes(scenario_cache_dir)
            row = {
                "scenario": name,
                "n_items": len(items),
                "n_bytes": n_bytes,
                "elapsed_s": round(elapsed, 2),
                "mb_s": round(n_bytes / elapsed / 1e6, 2),
            }
            writer.writerow(row)
            f.flush()
            log.info(row)
            progress.advance(task)

    log.info(f"download results written to {download_csv}")
    return local_items


def run_load(
    local_items: dict[Scenario, list],
    bands: list[str],
    chunk_sizes: list[int],
    worker_counts: list[int],
) -> None:
    """Sweep chunk size / worker count over `gfetch.mosaic.load()`.

    Parameters
    ----------
    local_items : dict[Scenario, list]
        Downloaded items per scenario name, with asset hrefs already local.
    bands : list[str]
        Asset keys to load.
    chunk_sizes : list[int]
        Dask chunk sizes (pixels, square) to sweep.
    worker_counts : list[int]
        Dask thread-pool sizes to sweep.
    """
    load_csv = RESULTS_DIR / "gfetch_pipeline_load.csv"
    fieldnames = ["scenario", "chunk", "num_workers", "n_bytes", "elapsed_s", "mpix_s", "mb_s"]

    with (
        load_csv.open("w", newline="") as f,
        sweep_progress() as progress,
    ):
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        n_runs = len(local_items) * len(chunk_sizes) * len(worker_counts)
        task = progress.add_task("gfetch load", total=n_runs)

        for name, items in local_items.items():
            bbox = WIDE_BBOX if name == "wide" else DEEP_BBOX
            geobox = GeoBox.from_bbox(bbox, crs="utm", resolution=RESOLUTION)

            for chunk in chunk_sizes:
                ds = load(items, geobox, bands, chunks={"x": chunk, "y": chunk})
                n_bytes_total = sum(v.nbytes for v in ds.data_vars.values())
                npix_total = ds.sizes["x"] * ds.sizes["y"] * ds.sizes["time"] * len(bands)

                for num_workers in worker_counts:
                    log.info(f"[{name}] chunk={chunk} num_workers={num_workers}: starting")
                    description = f"[{name}] chunk={chunk} workers={num_workers} load"
                    with (
                        dask.config.set(scheduler="threads", num_workers=num_workers),
                        dask_load_progress(progress, description),
                    ):
                        start = time.perf_counter()
                        ds.compute()
                        elapsed = time.perf_counter() - start

                    row = {
                        "scenario": name,
                        "chunk": chunk,
                        "num_workers": num_workers,
                        "n_bytes": n_bytes_total,
                        "elapsed_s": round(elapsed, 2),
                        "mpix_s": round(npix_total / elapsed / 1e6, 4),
                        "mb_s": round(n_bytes_total / elapsed / 1e6, 2),
                    }
                    writer.writerow(row)
                    f.flush()
                    log.info(row)
                    progress.advance(task)

    log.info(f"load results written to {load_csv}")


@app.default
def main(
    scenarios: list[Scenario] = DEFAULT_SCENARIOS,
    download_concurrency: int = DEFAULT_DOWNLOAD_CONCURRENCY,
    load_chunk_sizes: list[int] = DEFAULT_LOAD_CHUNK_SIZES,
    load_worker_counts: list[int] = DEFAULT_LOAD_WORKER_COUNTS,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    verbose: bool = False,
) -> None:
    """Benchmark gfetch's real search -> download -> mosaic pipeline.

    No S3/GDAL tuning is done in this script itself - `gfetch.mosaic.load()`
    now calls `odc.stac.configure_s3_access()` internally, so this checks that
    default, unassisted, actually reaches comparable throughput.

    Parameters
    ----------
    scenarios : list[Scenario]
        Scenarios to benchmark (see `common.py`). Defaults to `DEFAULT_SCENARIOS`
        (both).
    download_concurrency : int
        `max_concurrent_items` passed to `gfetch.download.download_items()`.
        Defaults to `DEFAULT_DOWNLOAD_CONCURRENCY` (32, the best concurrency found
        by `two_stage_mosaic.py`'s own sweep).
    load_chunk_sizes : list[int]
        Dask chunk sizes (pixels, square) to sweep in the load stage. Defaults to
        `DEFAULT_LOAD_CHUNK_SIZES`.
    load_worker_counts : list[int]
        Dask thread-pool sizes to sweep in the load stage. Defaults to
        `DEFAULT_LOAD_WORKER_COUNTS`.
    cache_dir : Path
        Directory to download into. Defaults to `DEFAULT_CACHE_DIR`, a dedicated
        subdirectory kept separate from `two_stage_mosaic.py`'s own cache so this
        script measures a real (cold) download rather than a resumed no-op.
    verbose : bool
        Log at DEBUG instead of INFO. Defaults to False.
    """
    setup_logging(verbose)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    profile = get_profile("sentinel-2")
    bands = list(profile.default_bands)

    scenario_items = {name: _search_scenario(name) for name in scenarios}
    for name, items in scenario_items.items():
        log.info(f"[{name}] {len(items)} items")

    local_items = run_download(scenario_items, cache_dir, bands, download_concurrency)
    run_load(local_items, bands, load_chunk_sizes, load_worker_counts)

    log.info("gfetch pipeline benchmark complete")


# %%
if __name__ == "__main__":
    app()
