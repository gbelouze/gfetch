import logging
from pathlib import Path

import dask
import pystac

from gfetch.cli.config import load
from gfetch.mosaic import group_by_utm_zone, zone_geobox
from gfetch.mosaic import mosaic as build_mosaic
from gfetch.profiles import get_profile
from gfetch.utils.progress import count_bar, dask_progress, temporary_task
from gfetch.utils.system import available_cpus
from gfetch.write import write

log = logging.getLogger(__name__)


def mosaic(config_path: Path) -> None:
    """
    Load, cloud-mask, and composite this job's items into a mosaic per UTM zone the
    AOI spans, and write each zone's mosaic to its own output Zarr store.

    Runs against the local cache if `gfetch download` has been run, otherwise loads
    directly from the items' remote hrefs (fine for a single internet-connected
    machine; a compute-only HPC node needs the local cache).

    Parameters
    ----------
    config_path : Path
        Path to the configuration YAML file.
    """
    cfg = load(config_path)

    if cfg.cached_items_path.exists():
        items_path = cfg.cached_items_path
    elif cfg.items_path.exists():
        items_path = cfg.items_path
        log.warning("No downloaded cache found - loading directly from remote hrefs.")
    else:
        log.error(f"{cfg.items_path} not found - run `gfetch search` first.")
        return

    items = list(pystac.ItemCollection.from_file(items_path))
    log.debug(f"Loaded {len(items)} item(s) from {items_path}")
    profile = get_profile(cfg.satellite)
    bands = list(cfg.bands) if cfg.bands else list(profile.default_bands)

    zones = group_by_utm_zone(items, cfg.aoi.bbox)
    log.info(f"Items span {len(zones)} UTM zone(s): {[crs.epsg for crs in zones]}")

    n_compute_workers = cfg.n_compute_workers or available_cpus()
    log.info(f"Using {n_compute_workers} dask worker thread(s)")

    with (
        count_bar() as progress,
        temporary_task(progress, "Computing zones", total=len(zones)) as zones_task,
        dask.config.set(num_workers=n_compute_workers),
    ):
        for crs, zone_items in zones.items():
            log.info(f"Computing and writing mosaic for {crs}...")
            ds = build_mosaic(
                zone_items,
                zone_geobox(crs, cfg.aoi.bbox, cfg.resolution),
                bands,
                mask_band=profile.cloud_mask_band,
                mask_out=profile.cloud_mask_out,
                resampling=cfg.resampling,
                chunks=cfg.chunks,
            )
            # `ds` stays lazy (dask-backed) all the way into `write()` - streaming
            # straight to Zarr computes and writes one chunk at a time via
            # `dask.array.store()`, so peak memory is bounded by chunk size x
            # concurrent workers, never the whole zone's materialized size. Calling
            # `.compute()` here first would force the entire zone into memory at
            # once regardless of chunk size, defeating `n_compute_workers` entirely
            # on a large AOI.
            with dask_progress(progress, f"{crs}"):
                write(ds, cfg.zarr_path(crs))
            progress.advance(zones_task)
