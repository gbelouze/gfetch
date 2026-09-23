import logging
from pathlib import Path

import pystac

from gfetch.cli.config import Config, load
from gfetch.search import search as search_items
from gfetch.sources import get_source

log = logging.getLogger(__name__)

_ORBIT_STATES = {"ascending", "descending"}


def _build_query(cfg: Config) -> dict | None:
    """
    Translate a config's optional search filters into a STAC query-extension dict.

    Parameters
    ----------
    cfg : Config
        Job configuration.

    Returns
    -------
    dict | None
        Query-extension filters, or None if none are set.

    Raises
    ------
    ValueError
        If `cfg.orbit_state` is set to anything other than 'ascending'/'descending'.
    """
    query: dict = {}
    if cfg.max_cloud_cover is not None:
        query["eo:cloud_cover"] = {"lt": cfg.max_cloud_cover}
    if cfg.orbit_state is not None:
        if cfg.orbit_state not in _ORBIT_STATES:
            raise ValueError(f"orbit_state must be one of {_ORBIT_STATES}, got {cfg.orbit_state!r}")
        query["sat:orbit_state"] = {"eq": cfg.orbit_state}
    return query or None


def search(config_path: Path, satellite_key: str) -> None:
    """
    Search a STAC source for items matching a configuration's AOI/time range, and
    write the results as this job's `search` -> `download` hand-off file.

    Parameters
    ----------
    config_path : Path
        Path to the configuration YAML file.
    satellite_key : str
        Which satellite to load from `config_path` - `'s1'`, `'s2'`, or a name under
        its `custom:` section. See `gfetch.cli.config.load`.
    """
    cfg = load(config_path, satellite_key)
    source = get_source(cfg.source)
    query = _build_query(cfg)
    log.debug(f"query={query}")

    intersects = None
    if cfg.countries is not None:
        import shapely

        from gfetch.countries import resolve_country_polygon

        intersects = shapely.geometry.mapping(resolve_country_polygon(cfg.countries))

    items = search_items(
        source,
        cfg.satellite,
        cfg.resolved_aoi.bbox,
        cfg.time_range.datetime,
        query=query,
        collection=cfg.collection,
        intersects=intersects,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pystac.ItemCollection(items).save_object(str(cfg.items_path))
    log.info(f"Wrote {len(items)} items to {cfg.items_path}")
