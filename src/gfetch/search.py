"""STAC search stage: given an AOI/time range/satellite, find matching STAC items."""

import logging

import pystac
import pystac_client

from gfetch.sources import StacSource

log = logging.getLogger(__name__)

__all__ = ["search"]


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
    log.info(f"Found {len(items)} items")
    return items
