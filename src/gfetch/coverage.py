"""Coverage stage: how many items the mosaic stage will composite in each shard.

Informational only, and needs no pixel data: runs offline against the `search` stage's
items. The shard grid is recomputed from the zone's geobox rather than read from the
Zarr store, which only exists once the mosaic stage has started, so it could drift from
the store's if the two ever disagree.
"""

import datetime
import itertools
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import geopandas as gpd
import pystac
from odc.geo.crs import CRS
from odc.geo.geobox import GeoBox

from gfetch.mosaic import ORBIT_STATES, FootprintIndex

log = logging.getLogger(__name__)

__all__ = ["coverage", "shard_regions", "solar_day"]


def shard_regions(shape: tuple[int, int], unit: dict[str, int]) -> list[dict[str, slice]]:
    """
    Tile a 2D pixel grid into storage units, as `gfetch.write.write_regions` lists them.

    Parameters
    ----------
    shape : tuple[int, int]
        Grid size, as `(y, x)`.
    unit : dict[str, int]
        Storage unit size along `"y"` and `"x"`: the shard size, or the chunk size
        when unsharded.

    Returns
    -------
    list[dict[str, slice]]
        Every unit's `{"y": slice, "x": slice}` region, in row-major order, edge
        units clipped to `shape`.
    """
    per_dim = [
        [slice(start, min(start + unit[dim], size)) for start in range(0, size, unit[dim])]
        for dim, size in zip(("y", "x"), shape, strict=True)
    ]
    return [{"y": y, "x": x} for y, x in itertools.product(*per_dim)]


def solar_day(item: pystac.Item, lon: float) -> datetime.date:
    """
    Date odc-stac's `groupby="solar_day"` assigns an item when loading at `lon`.

    Mirrors odc-stac's own rule: the item's nominal datetime (`datetime`, else
    `start_datetime`, else `end_datetime`) shifted by whole hours of solar time, at
    the loaded geobox's centroid longitude rather than the item's own.

    Parameters
    ----------
    item : pystac.Item
        Item to date.
    lon : float
        Centroid longitude of the geobox the item is loaded onto, in degrees.

    Returns
    -------
    datetime.date
        The item's solar day at `lon`.

    Raises
    ------
    ValueError
        If `item` has none of `datetime`, `start_datetime` or `end_datetime`.
    """
    candidates = (
        item.datetime,
        item.common_metadata.start_datetime,
        item.common_metadata.end_datetime,
    )
    nominal = next((ts for ts in candidates if ts is not None), None)
    if nominal is None:
        msg = f"Item {item.id} has no datetime, start_datetime or end_datetime"
        log.error(msg)
        raise ValueError(msg)
    return (nominal + datetime.timedelta(hours=int(lon / 15))).date()


def _shard_row(
    items: list[pystac.Item], geobox: GeoBox, *, split_orbit_states: bool
) -> dict[str, Any]:
    """
    Count what one shard's mosaic loads.

    Parameters
    ----------
    items : list[pystac.Item]
        Items intersecting the shard.
    geobox : GeoBox
        The shard's pixel grid.
    split_orbit_states : bool
        Also count items per `sat:orbit_state`.

    Returns
    -------
    dict[str, Any]
        `n_items`, `n_timesteps`, and with `split_orbit_states`, one `n_{state}` per
        state in `ORBIT_STATES`.
    """
    ((lon, _),) = geobox.extent.centroid.to_crs("EPSG:4326").points
    row: dict[str, Any] = {
        "n_items": len(items),
        "n_timesteps": len({solar_day(item, lon) for item in items}),
    }
    if split_orbit_states:
        for state in ORBIT_STATES:
            row[f"n_{state}"] = sum(
                item.properties.get("sat:orbit_state") == state for item in items
            )
    return row


def coverage(
    zones: Mapping[CRS, tuple[GeoBox, Sequence[pystac.Item]]],
    unit: dict[str, int],
    *,
    skip: Callable[[CRS, GeoBox], Callable[[dict[str, slice]], bool]] | None = None,
    split_orbit_states: bool = False,
) -> gpd.GeoDataFrame:
    """
    Count, per shard, the items the mosaic stage composites there.

    Items are selected per shard exactly as `gfetch.mosaic.mosaic` does, see
    `gfetch.mosaic.FootprintIndex`. A shard no item intersects gets zero counts
    (`mosaic` then loads one all-nodata time step).

    Parameters
    ----------
    zones : Mapping[CRS, tuple[GeoBox, Sequence[pystac.Item]]]
        Each UTM zone's output grid and items, as from `gfetch.mosaic.zone_geobox`
        and `gfetch.mosaic.group_by_utm_zone`.
    unit : dict[str, int]
        Storage unit size along `"y"` and `"x"`, see `shard_regions`.
    skip : Callable[[CRS, GeoBox], Callable[[dict[str, slice]], bool]] | None
        Builds, for a zone, the test for which of its shards `mosaic` never computes
        (e.g. from `gfetch.mosaic.outside_aoi`). Defaults to None (no shard skipped).
    split_orbit_states : bool
        Also count items per `sat:orbit_state`, as `mosaic(split_orbit_states=True)`
        composites each separately. Defaults to False.

    Returns
    -------
    gpd.GeoDataFrame
        One row per shard, in EPSG:4326: `epsg`, `y`, `x` (the shard's zone and pixel
        offset), `skipped`, the counts (see `_shard_row`), and the shard's extent as
        `geometry`.
    """
    rows = []
    for crs, (geobox, items) in zones.items():
        index = FootprintIndex(items)
        is_skipped = skip(crs, geobox) if skip is not None else None
        regions = shard_regions(geobox.shape.yx, unit)
        log.debug(f"EPSG:{crs.epsg}: {len(regions)} shard(s), {len(items)} item(s)")
        for region in regions:
            shard = cast("GeoBox", geobox[region["y"], region["x"]])
            rows.append(
                {
                    "epsg": crs.epsg,
                    "y": region["y"].start,
                    "x": region["x"].start,
                    "skipped": is_skipped(region) if is_skipped is not None else False,
                    **_shard_row(
                        index.intersecting(shard), shard, split_orbit_states=split_orbit_states
                    ),
                    "geometry": shard.extent.to_crs("EPSG:4326", wrapdateline=True).geom,
                }
            )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
