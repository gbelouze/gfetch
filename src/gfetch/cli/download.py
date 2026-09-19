import asyncio
import logging
from pathlib import Path

import pystac

from gfetch.cli.config import load
from gfetch.download import download_items
from gfetch.profiles import get_profile
from gfetch.utils.progress import default_bar

log = logging.getLogger(__name__)


def download(config_path: Path) -> None:
    """
    Download the assets of this job's searched items into the local cache, and write
    the local-href items as this job's `download` -> `mosaic` hand-off file.

    Parameters
    ----------
    config_path : Path
        Path to the configuration YAML file.
    """
    cfg = load(config_path)
    if not cfg.items_path.exists():
        log.error(f"{cfg.items_path} not found - run `gfetch search` first.")
        return

    items = list(pystac.ItemCollection.from_file(cfg.items_path))
    profile = get_profile(cfg.satellite)
    asset_keys = list(cfg.bands) if cfg.bands else list(profile.default_bands)
    if profile.cloud_mask_band is not None and profile.cloud_mask_band not in asset_keys:
        asset_keys.append(profile.cloud_mask_band)

    with default_bar() as progress:
        cached_items = asyncio.run(
            download_items(
                items,
                cfg.cache_dir,
                asset_keys,
                max_concurrent_items=cfg.n_workers,
                progress=progress,
            )
        )

    pystac.ItemCollection(cached_items).save_object(str(cfg.cached_items_path))
    log.info(f"Downloaded {len(cached_items)} items into {cfg.cache_dir}")
