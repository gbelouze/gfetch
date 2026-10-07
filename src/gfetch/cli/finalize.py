import logging
from pathlib import Path

from gfetch.cli.config import load
from gfetch.cli.stores import zone_stores
from gfetch.finalize import remove_cache
from gfetch.utils.progress import count_bar
from gfetch.vrt import write_vrt
from gfetch.write import store_initialized

log = logging.getLogger(__name__)


def vrt(config_path: Path, satellite_key: str) -> None:
    """
    Write a multi-band VRT next to each of this job's Zarr stores.

    Stores that don't exist yet are skipped, with a warning.

    Parameters
    ----------
    config_path : Path
        Path to the configuration YAML file.
    satellite_key : str
        Which satellite to load from `config_path` - `'s1'`, `'s2'`, or a name under
        its `custom:` section. See `gfetch.cli.config.load`.
    """
    cfg = load(config_path, satellite_key)
    if not cfg.items_path.exists():
        log.error(f"{cfg.items_path} not found, run `gfetch search` first.")
        return
    for store in zone_stores(cfg):
        if not store_initialized(store):
            log.warning(f"{store} not found, run `gfetch mosaic` first. Skipping.")
            continue
        write_vrt(store)


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
    if not cfg.items_path.exists():
        log.error(f"{cfg.items_path} not found, run `gfetch search` first.")
        return
    with count_bar() as progress:
        remove_cache(
            cfg.cache_dir,
            zone_stores(cfg),
            cached_items_path=cfg.cached_items_path,
            progress=progress,
        )
