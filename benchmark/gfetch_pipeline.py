"""Benchmark gfetch's own `search`/`download`/`mosaic` functions end-to-end on the
"wide"/"deep" scenarios also used by this directory's other, notebook-style
benchmarks (concurrency.py/chunks.py/two_stage_mosaic.py), across one or more STAC
sources - the goal is to check whether the actual library, now that odc.stac.load()'s
S3-credential fix and search()'s processing-baseline dedup have been folded back in
(see claude/tech-stack.md's Decision log), reaches throughput comparable to the
reference numbers those other scripts measured, rather than reimplementing the same
download/load logic by hand.

**2026-09-22: added Planetary Computer alongside Earth Search** (`--sources`), per
`claude/tech-stack.md`'s "Next steps" item to actually exercise the PC source -
confirmed live that PC needs no authentication at all (`stac_asset`'s
`PlanetaryComputerClient` fetches its SAS-signing token from PC's public endpoint
anonymously, same as Earth Search needs no signing at all), but its `sentinel-2-l2a`
collection uses different asset keys/properties than Earth Search's
`sentinel-2-c1-l2a` (see `common.py::SOURCE_BANDS`/`SOURCE_DEEP_TILE_PROPERTY`) - a
gfetch satellite profile's `default_bands` ("red"/"green"/"blue") silently downloads
zero bytes against PC, since those asset keys don't exist on its items. This script
works around that with its own per-source band list; `gfetch.profiles.PROFILES`
itself is unchanged for now (open question, see `claude/tech-stack.md`'s "Planetary
Computer: asset-key divergence" entry).

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
    RESOLUTION,
    SOURCE_BANDS,
    SOURCE_DEEP_TILE_PROPERTY,
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
from gfetch.search import search
from gfetch.sources import get_source

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).parent / "cache" / "gfetch_pipeline"
RESULTS_DIR = Path(__file__).parent / "results"

DEFAULT_SCENARIOS: list[Scenario] = ["wide", "deep"]
DEFAULT_SOURCES: list[str] = ["earthsearch"]
DEFAULT_DOWNLOAD_CONCURRENCY = 32
DEFAULT_LOAD_CHUNK_SIZES = [512, 2048, 7168]
DEFAULT_LOAD_WORKER_COUNTS = [dask.system.CPU_COUNT, 32]

app = cyclopts.App()


def _search_scenario(scenario: Scenario, source_name: str) -> list:
    """
    Search gfetch's own `search()` for one scenario's AOI/date range, on one source.

    Parameters
    ----------
    scenario : Scenario
        `"wide"` or `"deep"` (see `common.py`).
    source_name : str
        STAC source name (e.g. `"earthsearch"`, `"planetary-computer"`).

    Returns
    -------
    list[pystac.Item]
        Matching items - already deduped by processing baseline, since that now
        happens inside `gfetch.search.search()` itself.
    """
    source = get_source(source_name)
    if scenario == "wide":
        return search(source, "sentinel-2", bbox=WIDE_BBOX, datetime=WIDE_DATETIME)
    items = search(source, "sentinel-2", bbox=DEEP_BBOX, datetime=DEEP_DATETIME)
    prop_key, expected = SOURCE_DEEP_TILE_PROPERTY[source_name]
    return [it for it in items if it.properties.get(prop_key) == expected]


def _cache_size_bytes(cache_dir: Path) -> int:
    """
    Total size in bytes of every file under `cache_dir`.

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
    source_scenario_items: dict[str, dict[Scenario, list]],
    cache_dir: Path,
    max_concurrency: int,
) -> dict[str, dict[Scenario, list]]:
    """
    Download each (source, scenario)'s items via `gfetch.download.download_items()`.

    Parameters
    ----------
    source_scenario_items : dict[str, dict[Scenario, list]]
        Search-result items per source name, per scenario name.
    cache_dir : Path
        Base directory to download into (one subdirectory per source/scenario name).
    max_concurrency : int
        `max_concurrent_items` passed through to `download_items()`.

    Returns
    -------
    dict[str, dict[Scenario, list]]
        Downloaded items per source name, per scenario name, with asset hrefs
        rewritten to local paths.
    """
    download_csv = RESULTS_DIR / "gfetch_pipeline_download.csv"
    local_items: dict[str, dict[Scenario, list]] = {}
    total = sum(len(scenario_items) for scenario_items in source_scenario_items.values())

    with (
        download_csv.open("w", newline="") as f,
        sweep_progress() as progress,
    ):
        writer = csv.DictWriter(
            f, fieldnames=["source", "scenario", "n_items", "n_bytes", "elapsed_s", "mb_s"]
        )
        writer.writeheader()
        task = progress.add_task("gfetch download", total=total)

        for source_name, scenario_items in source_scenario_items.items():
            bands = list(SOURCE_BANDS[source_name])
            local_items[source_name] = {}

            for name, items in scenario_items.items():
                scenario_cache_dir = cache_dir / source_name / name
                log.info(
                    f"[{source_name}/{name}] downloading {len(items)} items "
                    f"(bands={bands}) at max_concurrency={max_concurrency}"
                )
                start = time.perf_counter()
                local_items[source_name][name] = asyncio.run(
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
                    "source": source_name,
                    "scenario": name,
                    "n_items": len(items),
                    "n_bytes": n_bytes,
                    "elapsed_s": round(elapsed, 2),
                    "mb_s": round(n_bytes / elapsed / 1e6, 2) if elapsed else 0.0,
                }
                writer.writerow(row)
                f.flush()
                log.info(row)
                progress.advance(task)

    log.info(f"download results written to {download_csv}")
    return local_items


def run_load(
    source_local_items: dict[str, dict[Scenario, list]],
    chunk_sizes: list[int],
    worker_counts: list[int],
) -> None:
    """
    Sweep chunk size / worker count over `gfetch.mosaic.load()`, per source/scenario.

    Parameters
    ----------
    source_local_items : dict[str, dict[Scenario, list]]
        Downloaded items per source name, per scenario name, with asset hrefs
        already local.
    chunk_sizes : list[int]
        Dask chunk sizes (pixels, square) to sweep.
    worker_counts : list[int]
        Dask thread-pool sizes to sweep.
    """
    load_csv = RESULTS_DIR / "gfetch_pipeline_load.csv"
    fieldnames = [
        "source",
        "scenario",
        "chunk",
        "num_workers",
        "n_bytes",
        "elapsed_s",
        "mpix_s",
        "mb_s",
    ]
    n_runs = (
        sum(len(scenario_items) for scenario_items in source_local_items.values())
        * len(chunk_sizes)
        * len(worker_counts)
    )

    with (
        load_csv.open("w", newline="") as f,
        sweep_progress() as progress,
    ):
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        task = progress.add_task("gfetch load", total=n_runs)

        for source_name, local_items in source_local_items.items():
            bands = list(SOURCE_BANDS[source_name])

            for name, items in local_items.items():
                bbox = WIDE_BBOX if name == "wide" else DEEP_BBOX
                geobox = GeoBox.from_bbox(bbox, crs="utm", resolution=RESOLUTION)

                for chunk in chunk_sizes:
                    ds = load(items, geobox, bands, chunks={"x": chunk, "y": chunk})
                    n_bytes_total = sum(v.nbytes for v in ds.data_vars.values())
                    npix_total = ds.sizes["x"] * ds.sizes["y"] * ds.sizes["time"] * len(bands)

                    for num_workers in worker_counts:
                        log.info(
                            f"[{source_name}/{name}] chunk={chunk} "
                            f"num_workers={num_workers}: starting"
                        )
                        description = (
                            f"[{source_name}/{name}] chunk={chunk} workers={num_workers} load"
                        )
                        with (
                            dask.config.set(scheduler="threads", num_workers=num_workers),
                            dask_load_progress(progress, description),
                        ):
                            start = time.perf_counter()
                            ds.compute()
                            elapsed = time.perf_counter() - start

                        row = {
                            "source": source_name,
                            "scenario": name,
                            "chunk": chunk,
                            "num_workers": num_workers,
                            "n_bytes": n_bytes_total,
                            "elapsed_s": round(elapsed, 2),
                            "mpix_s": round(npix_total / elapsed / 1e6, 4) if elapsed else 0.0,
                            "mb_s": round(n_bytes_total / elapsed / 1e6, 2) if elapsed else 0.0,
                        }
                        writer.writerow(row)
                        f.flush()
                        log.info(row)
                        progress.advance(task)

    log.info(f"load results written to {load_csv}")


@app.default
def main(
    scenarios: list[Scenario] = DEFAULT_SCENARIOS,
    sources: list[str] = DEFAULT_SOURCES,
    download_concurrency: int = DEFAULT_DOWNLOAD_CONCURRENCY,
    load_chunk_sizes: list[int] = DEFAULT_LOAD_CHUNK_SIZES,
    load_worker_counts: list[int] = DEFAULT_LOAD_WORKER_COUNTS,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    verbose: bool = False,
) -> None:
    """
    Benchmark gfetch's real search -> download -> mosaic pipeline, across sources.

    No S3/GDAL tuning is done in this script itself - `gfetch.mosaic.load()`
    now calls `odc.stac.configure_s3_access()` internally, so this checks that
    default, unassisted, actually reaches comparable throughput.

    Parameters
    ----------
    scenarios : list[Scenario]
        Scenarios to benchmark (see `common.py`). Defaults to `DEFAULT_SCENARIOS`
        (both).
    sources : list[str]
        STAC source names to benchmark (e.g. `"earthsearch"`,
        `"planetary-computer"`). Defaults to `DEFAULT_SOURCES` (`earthsearch` only -
        pass both explicitly to compare).
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

    source_scenario_items = {
        source_name: {name: _search_scenario(name, source_name) for name in scenarios}
        for source_name in sources
    }
    for source_name, scenario_items in source_scenario_items.items():
        for name, items in scenario_items.items():
            log.info(f"[{source_name}/{name}] {len(items)} items")

    source_local_items = run_download(source_scenario_items, cache_dir, download_concurrency)
    run_load(source_local_items, load_chunk_sizes, load_worker_counts)

    log.info("gfetch pipeline benchmark complete")


# %%
if __name__ == "__main__":
    app()
