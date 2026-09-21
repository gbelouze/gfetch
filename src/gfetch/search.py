"""STAC search stage: given an AOI/time range/satellite, find matching STAC items."""

import logging

import pystac
import pystac_client

from gfetch.sources import StacSource

log = logging.getLogger(__name__)

__all__ = ["search"]


def _dedupe_sentinel2_processing_baseline(items: list[pystac.Item]) -> list[pystac.Item]:
    """
    Keep one item per (grid tile, date), preferring the highest `s2:processing_baseline`.

    Earth Search lists the same tile/date twice when ESA reprocesses the archive to a
    new baseline, keeping the superseded item rather than removing it. Loading both
    wastes bandwidth (`groupby="solar_day"` only discards the duplicate after both have
    already been fetched), and blending items across baselines in one composite is a
    radiometric correctness risk: baseline 04.00 changed how DN values encode
    reflectance for negative values.

    Parameters
    ----------
    items : list[pystac.Item]
        Search results, possibly containing multiple processing baselines per (tile,
        date).

    Returns
    -------
    list[pystac.Item]
        One item per (tile, date): the highest `s2:processing_baseline` of any
        candidates sharing that key.
    """
    best: dict[tuple[str, object], pystac.Item] = {}
    for item in items:
        assert item.datetime is not None
        key = (item.properties.get("grid:code", item.id), item.datetime.date())
        baseline = item.properties.get("s2:processing_baseline", "0")
        if key not in best or baseline > best[key].properties.get("s2:processing_baseline", "0"):
            best[key] = item
    return list(best.values())


def search(
    source: StacSource,
    satellite: str,
    bbox: tuple[float, float, float, float],
    datetime: str,
    max_items: int | None = None,
    query: dict | None = None,
) -> list[pystac.Item]:
    """
    Search a STAC source for items matching an AOI and time range.

    Parameters
    ----------
    source : StacSource
        STAC API source to search against.
    satellite : str
        Satellite/profile name (e.g. 'sentinel-2'), resolved to a collection id via
        `source`.
    bbox : tuple[float, float, float, float]
        Bounding box (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
    datetime : str
        Date/time range in STAC API format (e.g. '2024-01-01/2024-06-01').
    max_items : int | None
        Maximum number of items to return. Defaults to None (no limit).
    query : dict | None
        Additional STAC API query-extension filters (e.g. cloud cover). Defaults to
        None.

    Returns
    -------
    list[pystac.Item]
        Matching STAC items.
    """
    collection = source.collection(satellite)
    log.info(
        f"Searching {source.name} collection {collection!r} for bbox={bbox} datetime={datetime}"
    )
    catalog = pystac_client.Client.open(source.api_url)
    result = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=datetime,
        query=query,
        max_items=max_items,
    )
    items = list(result.items())
    if satellite == "sentinel-2":
        deduped = _dedupe_sentinel2_processing_baseline(items)
        if len(deduped) != len(items):
            log.info(
                f"Dropped {len(items) - len(deduped)} superseded-processing-baseline duplicate(s)"
            )
        items = deduped
    log.info(f"Found {len(items)} items")
    return items
