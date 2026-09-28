import logging
from pathlib import Path

import pystac

from gfetch.cli.config import ORBIT_STATE_AS_BANDS, load
from gfetch.coverage import coverage as build_coverage
from gfetch.mosaic import (
    group_by_utm_zone,
    outside_aoi,
    resolve_chunks,
    resolve_shards,
    zone_geobox,
)
from gfetch.utils.geoparquet import write_geoparquet

log = logging.getLogger(__name__)


def coverage(config_path: Path, satellite_key: str) -> None:
    """
    Write a GeoParquet of how many items `mosaic` will composite in each shard.

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

    items = list(pystac.ItemCollection.from_file(cfg.items_path))
    bbox = cfg.resolved_aoi.bbox
    zones = {
        crs: (zone_geobox(crs, bbox, cfg.resolution), zone_items)
        for crs, zone_items in group_by_utm_zone(items, bbox).items()
    }
    chunks = resolve_chunks(cfg.chunks)
    shards = resolve_shards(cfg.shard_factor, chunks)
    gdf = build_coverage(
        zones,
        shards if shards is not None else chunks,
        skip=lambda _crs, geobox: outside_aoi(geobox, cfg.aoi_geometry),
        split_orbit_states=cfg.orbit_state == ORBIT_STATE_AS_BANDS,
    )
    write_geoparquet(gdf, cfg.coverage_path)

    computed = gdf[~gdf["skipped"]]
    log.info(
        f"Wrote {len(gdf)} shard(s) to {cfg.coverage_path}, {len(computed)} of them "
        f"computed by `mosaic`, {int((computed['n_items'] == 0).sum())} of those with no item"
    )
