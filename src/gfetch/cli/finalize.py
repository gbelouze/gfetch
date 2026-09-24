import logging
from pathlib import Path

import pystac

from gfetch.cli.config import Config, load, resolve_output_variables
from gfetch.finalize import pack_store, remove_cache
from gfetch.mosaic import group_by_utm_zone
from gfetch.utils.progress import count_bar, default_bar

log = logging.getLogger(__name__)


def _zone_stores(cfg: Config) -> list[Path] | None:
    """
    List the Zarr stores `mosaic` writes for a job, one per UTM zone its items span.

    Parameters
    ----------
    cfg : Config
        Job configuration.

    Returns
    -------
    list[Path] | None
        Store paths, or None if the job's search results don't exist.
    """
    if not cfg.items_path.exists():
        log.error(f"{cfg.items_path} not found, run `gfetch search` first.")
        return None
    items = list(pystac.ItemCollection.from_file(cfg.items_path))
    return [cfg.zarr_path(crs) for crs in group_by_utm_zone(items, cfg.resolved_aoi.bbox)]


def pack(config_path: Path, satellite_key: str, *, remove_store: bool = False) -> None:
    """
    Pack each of this job's complete Zarr stores into a single-file zip store.

    Parameters
    ----------
    config_path : Path
        Path to the configuration YAML file.
    satellite_key : str
        Which satellite to load from `config_path` - `'s1'`, `'s2'`, or a name under
        its `custom:` section. See `gfetch.cli.config.load`.
    remove_store : bool
        Delete each store directory once its zip is in place. Defaults to False.
    """
    cfg = load(config_path, satellite_key)
    stores = _zone_stores(cfg)
    if stores is None:
        return
    variables = resolve_output_variables(cfg)
    with default_bar() as progress:
        for store in stores:
            pack_store(store, variables, remove_source=remove_store, progress=progress)


def clean(config_path: Path, satellite_key: str) -> None:
    """
    Delete this job's download cache once every one of its Zarr stores is complete.

    Parameters
    ----------
    config_path : Path
        Path to the configuration YAML file.
    satellite_key : str
        Which satellite to load from `config_path` - `'s1'`, `'s2'`, or a name under
        its `custom:` section. See `gfetch.cli.config.load`.
    """
    cfg = load(config_path, satellite_key)
    stores = _zone_stores(cfg)
    if stores is None:
        return
    with count_bar() as progress:
        remove_cache(
            cfg.cache_dir,
            stores,
            resolve_output_variables(cfg),
            cached_items_path=cfg.cached_items_path,
            progress=progress,
        )
