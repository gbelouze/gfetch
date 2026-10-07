import logging
from pathlib import Path

import pystac

from gfetch.cli.config import Config
from gfetch.mosaic import group_by_utm_zone
from gfetch.write import STACKED_VARIABLE, store_is_complete

log = logging.getLogger(__name__)


def zone_stores(cfg: Config) -> list[Path]:
    """
    List the Zarr stores `mosaic` writes for a job, one per UTM zone its items span.

    Parameters
    ----------
    cfg : Config
        Job configuration. Its search results (`cfg.items_path`) must exist.

    Returns
    -------
    list[Path]
        Store paths.
    """
    items = list(pystac.ItemCollection.from_file(cfg.items_path))
    return [cfg.zarr_path(crs) for crs in group_by_utm_zone(items, cfg.resolved_aoi.bbox)]


def stores_complete(cfg: Config) -> bool:
    """
    Check whether a previous run already wrote every one of a job's Zarr stores.

    The stores are listed from the job's existing search results, so a changed config
    goes unnoticed: the stores must be deleted to rerun the job.

    Parameters
    ----------
    cfg : Config
        Job configuration.

    Returns
    -------
    bool
        True if the job's search results exist, span at least one store, and every
        such store is complete (see `gfetch.write.store_is_complete`).
    """
    if not cfg.items_path.exists():
        return False
    stores = zone_stores(cfg)
    return bool(stores) and all(store_is_complete(s, [STACKED_VARIABLE]) for s in stores)
