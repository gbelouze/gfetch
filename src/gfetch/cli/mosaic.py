import logging
from pathlib import Path

import pystac

from gfetch.cli.config import load
from gfetch.mosaic import mosaic_by_zone
from gfetch.profiles import get_profile
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
    profile = get_profile(cfg.satellite)
    bands = list(cfg.bands) if cfg.bands else list(profile.default_bands)

    zone_datasets = mosaic_by_zone(
        items,
        cfg.aoi.bbox,
        bands,
        resolution=cfg.resolution,
        mask_band=profile.cloud_mask_band,
        mask_out=profile.cloud_mask_out,
    )

    for crs, ds in zone_datasets.items():
        log.info(f"Computing mosaic for {crs}...")
        computed = ds.compute()
        write(computed, cfg.zarr_path(crs))
