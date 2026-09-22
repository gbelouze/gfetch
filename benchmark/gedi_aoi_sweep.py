"""Benchmark `gfetch.gedi.fetch_gedi_l2a()` (SlideRule's on-demand GEDI L2A
subsetting) as the spatial and temporal AOI grow, independently.

Unlike `gfetch_pipeline.py`'s Sentinel-2 sweep, there's no download/load split to
benchmark separately: one call to `fetch_gedi_l2a()` both resolves the matching
granules (via CMR, server-side) and returns the already-subsetted result, so this
just measures that one call's wall-clock time and returned footprint count as each
axis grows on its own, with the other axis held fixed.

Depends on the `gfetch` package itself, like `gfetch_pipeline.py` - this calls the
real library function rather than reimplementing the SlideRule request by hand.
"""

# %%
import csv
import datetime
import logging
import math
import time
from pathlib import Path

import cyclopts
from common import setup_logging, sweep_progress

from gfetch.gedi import fetch_gedi_l2a

log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"

# Fontainebleau forest, France - same area used by gfetch's own live tests
# (tests/test_gedi.py), small enough to keep the smallest sweep cases fast.
CENTER_LON = 2.65
CENTER_LAT = 48.425

# GEDI's mission start (first L2A data available) - the temporal sweep anchors its
# start here and grows the end date, so each larger case's archive coverage is a
# superset of the smaller ones instead of an arbitrary unrelated window.
MISSION_START = "2019-04-17T00:00:00Z"

# Rough degrees-per-km at CENTER_LAT, for reporting area in km^2 alongside the raw
# degree bbox actually sent to SlideRule - footprint count/timing scale with true
# ground area, not degrees, and a degree of longitude shrinks away from the equator.
_KM_PER_DEG_LAT = 111.0
_KM_PER_DEG_LON = 111.32 * math.cos(math.radians(CENTER_LAT))

DEFAULT_SPATIAL_SIDES_DEG = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
DEFAULT_SPATIAL_TIME_START = "2020-01-01T00:00:00Z"
DEFAULT_SPATIAL_TIME_END = "2020-12-31T23:59:59Z"

DEFAULT_TEMPORAL_DAYS = [30, 90, 180, 365, 730, 1460]
DEFAULT_TEMPORAL_SIDE_DEG = 0.25

app = cyclopts.App()


def _square_bbox(side_deg: float) -> tuple[float, float, float, float]:
    """
    Build a square bbox of the given side length, centered on `CENTER_LON`/`CENTER_LAT`.

    Parameters
    ----------
    side_deg : float
        Side length in decimal degrees.

    Returns
    -------
    tuple[float, float, float, float]
        (min_lon, min_lat, max_lon, max_lat).
    """
    half = side_deg / 2
    return (
        CENTER_LON - half,
        CENTER_LAT - half,
        CENTER_LON + half,
        CENTER_LAT + half,
    )


def _run_case(
    bbox: tuple[float, float, float, float], time_range: tuple[str, str]
) -> tuple[int, float]:
    """
    Run and time one `fetch_gedi_l2a()` call.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Bounding box (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
    time_range : tuple[str, str]
        (t0, t1) passed straight through to `fetch_gedi_l2a()`.

    Returns
    -------
    tuple[int, float]
        (n_footprints, elapsed_s).
    """
    start = time.perf_counter()
    gdf = fetch_gedi_l2a(bbox, time_range=time_range)
    elapsed = time.perf_counter() - start
    return len(gdf), elapsed


def run_spatial_sweep(sides_deg: list[float], time_range: tuple[str, str]) -> None:
    """
    Sweep bbox side length at a fixed time range, writing `results/gedi_spatial_sweep.csv`.

    Parameters
    ----------
    sides_deg : list[float]
        Square bbox side lengths (decimal degrees) to sweep.
    time_range : tuple[str, str]
        (t0, t1) held fixed across every case in this sweep.
    """
    out_csv = RESULTS_DIR / "gedi_spatial_sweep.csv"
    fieldnames = ["side_deg", "area_km2", "n_footprints", "elapsed_s", "footprints_s"]

    with out_csv.open("w", newline="") as f, sweep_progress() as progress:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        task = progress.add_task("gedi spatial sweep", total=len(sides_deg))

        for side in sides_deg:
            bbox = _square_bbox(side)
            area_km2 = (side * _KM_PER_DEG_LON) * (side * _KM_PER_DEG_LAT)
            log.info(f"side={side}deg (~{area_km2:.0f}km^2) time_range={time_range}: starting")
            n_footprints, elapsed = _run_case(bbox, time_range)

            row = {
                "side_deg": side,
                "area_km2": round(area_km2, 1),
                "n_footprints": n_footprints,
                "elapsed_s": round(elapsed, 2),
                "footprints_s": round(n_footprints / elapsed, 1) if elapsed else 0.0,
            }
            writer.writerow(row)
            f.flush()
            log.info(row)
            progress.advance(task)

    log.info(f"spatial sweep results written to {out_csv}")


def run_temporal_sweep(days: list[int], side_deg: float) -> None:
    """
    Sweep time range length (anchored at `MISSION_START`) at a fixed bbox, writing
    `results/gedi_temporal_sweep.csv`.

    Parameters
    ----------
    days : list[int]
        Time range lengths, in days from `MISSION_START`, to sweep.
    side_deg : float
        Square bbox side length (decimal degrees) held fixed across every case.
    """
    out_csv = RESULTS_DIR / "gedi_temporal_sweep.csv"
    fieldnames = ["n_days", "n_footprints", "elapsed_s", "footprints_s"]
    bbox = _square_bbox(side_deg)
    mission_start_dt = datetime.datetime.strptime(MISSION_START, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.UTC
    )

    with out_csv.open("w", newline="") as f, sweep_progress() as progress:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        task = progress.add_task("gedi temporal sweep", total=len(days))

        for n_days in days:
            end_dt = mission_start_dt + datetime.timedelta(days=n_days)
            time_range = (MISSION_START, end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
            log.info(f"n_days={n_days} bbox={bbox}: starting")
            n_footprints, elapsed = _run_case(bbox, time_range)

            row = {
                "n_days": n_days,
                "n_footprints": n_footprints,
                "elapsed_s": round(elapsed, 2),
                "footprints_s": round(n_footprints / elapsed, 1) if elapsed else 0.0,
            }
            writer.writerow(row)
            f.flush()
            log.info(row)
            progress.advance(task)

    log.info(f"temporal sweep results written to {out_csv}")


@app.default
def main(
    spatial_sides_deg: list[float] = DEFAULT_SPATIAL_SIDES_DEG,
    spatial_time_start: str = DEFAULT_SPATIAL_TIME_START,
    spatial_time_end: str = DEFAULT_SPATIAL_TIME_END,
    temporal_days: list[int] = DEFAULT_TEMPORAL_DAYS,
    temporal_side_deg: float = DEFAULT_TEMPORAL_SIDE_DEG,
    verbose: bool = False,
) -> None:
    """
    Benchmark `fetch_gedi_l2a()` as spatial AOI size and temporal range grow, independently.

    Two independent sweeps, each holding the other axis fixed: growing a square bbox
    at a fixed one-year time range (`spatial_sides_deg`), and growing a time range
    anchored at GEDI's mission start at a fixed bbox (`temporal_days`). No
    timeout-enforcing subprocess per case (unlike the raster sweeps in this
    directory) - a SlideRule request is a plain blocking HTTP call, not a dask graph
    with no cancellation API, so a hung case is meant to be interrupted by hand.

    Parameters
    ----------
    spatial_sides_deg : list[float]
        Square bbox side lengths (decimal degrees) to sweep in the spatial sweep.
        Defaults to `DEFAULT_SPATIAL_SIDES_DEG`.
    spatial_time_start : str
        Time range start for the spatial sweep. Defaults to
        `DEFAULT_SPATIAL_TIME_START`.
    spatial_time_end : str
        Time range end for the spatial sweep. Defaults to `DEFAULT_SPATIAL_TIME_END`.
    temporal_days : list[int]
        Time range lengths, in days from `MISSION_START`, to sweep in the temporal
        sweep. Defaults to `DEFAULT_TEMPORAL_DAYS`.
    temporal_side_deg : float
        Square bbox side length (decimal degrees) held fixed in the temporal sweep.
        Defaults to `DEFAULT_TEMPORAL_SIDE_DEG`.
    verbose : bool
        Log at DEBUG instead of INFO. Defaults to False.
    """
    setup_logging(verbose)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_spatial_sweep(spatial_sides_deg, (spatial_time_start, spatial_time_end))
    run_temporal_sweep(temporal_days, temporal_side_deg)

    log.info("gedi aoi sweep benchmark complete")


# %%
if __name__ == "__main__":
    app()
