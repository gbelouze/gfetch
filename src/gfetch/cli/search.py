import logging
from pathlib import Path

import pystac

from gfetch.cli.config import load
from gfetch.search import search as search_items
from gfetch.sources import get_source

log = logging.getLogger(__name__)


def search(config_path: Path) -> None:
    """
    Search a STAC source for items matching a configuration's AOI/time range, and
    write the results as this job's `search` -> `download` hand-off file.

    Parameters
    ----------
    config_path : Path
        Path to the configuration YAML file.
    """
    cfg = load(config_path)
    source = get_source(cfg.source)
    query = (
        {"eo:cloud_cover": {"lt": cfg.max_cloud_cover}} if cfg.max_cloud_cover is not None else None
    )
    items = search_items(source, cfg.satellite, cfg.aoi.bbox, cfg.time_range.datetime, query=query)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pystac.ItemCollection(items).save_object(str(cfg.items_path))
    log.info(f"Wrote {len(items)} items to {cfg.items_path}")
