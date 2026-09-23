import asyncio
import logging
from pathlib import Path

import pystac

from gfetch.cli.config import load, resolve_bands, resolve_cloud_mask
from gfetch.download import download_items
from gfetch.utils.progress import default_bar

log = logging.getLogger(__name__)


def download(config_path: Path, satellite_key: str) -> None:
    """
    Download the assets of this job's searched items into the local cache, and write
    the local-href items as this job's `download` -> `mosaic` hand-off file.

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
        log.error(f"{cfg.items_path} not found - run `gfetch search` first.")
        return

    items = list(pystac.ItemCollection.from_file(cfg.items_path))
    mask_band, mask_out = resolve_cloud_mask(cfg)
    asset_keys = resolve_bands(cfg)
    if mask_band is not None and mask_band not in asset_keys:
        asset_keys.append(mask_band)
    log.debug(f"Resolved asset keys: {asset_keys}")

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
    log.info(f"Wrote {len(cached_items)} cached items to {cfg.cached_items_path}")
