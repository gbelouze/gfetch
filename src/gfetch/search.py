"""STAC search stage: given an AOI/time range/satellite, find matching STAC items."""

import datetime as _dt
import logging

import pystac
import pystac_client

from gfetch.sources import StacSource, resolve_collection

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


def _parse_date_range(
    datetime: str,
) -> tuple[_dt.date | None, _dt.date | None]:
    """
    Parse a STAC API datetime or datetime interval into its first and last days.

    Parameters
    ----------
    datetime : str
        A single datetime, or an interval `start/end` whose open ends are `..` or
        empty (e.g. '2024-01-01/2024-06-01', '2024-01-01T00:00:00Z/..').

    Returns
    -------
    tuple[datetime.date | None, datetime.date | None]
        First and last day, None for an open end.
    """
    start, _, end = datetime.partition("/") if "/" in datetime else (datetime, "", datetime)

    def _day(value: str) -> _dt.date | None:
        return None if value in ("", "..") else _dt.date.fromisoformat(value[:10])

    return _day(start), _day(end)


def search(
    source: StacSource,
    satellite: str,
    bbox: tuple[float, float, float, float],
    datetime: str,
    max_items: int | None = None,
    query: dict | None = None,
    collection: str | None = None,
    intersects: dict | None = None,
) -> list[pystac.Item]:
    """
    Search a STAC source for items matching an AOI and time range.

    Parameters
    ----------
    source : StacSource
        STAC API source to search against.
    satellite : str
        Satellite/profile name (e.g. 'sentinel-2'), resolved to a collection id via
        `source` unless `collection` is given directly.
    bbox : tuple[float, float, float, float]
        Bounding box (min_lon, min_lat, max_lon, max_lat) in EPSG:4326. Still used
        for logging even when `intersects` is given.
    datetime : str
        Date/time range in STAC API format (e.g. '2024-01-01/2024-06-01').
    max_items : int | None
        Maximum number of items to return. Defaults to None (no limit).
    query : dict | None
        Additional STAC API query-extension filters (e.g. cloud cover). Defaults to
        None.
    collection : str | None
        Explicit STAC collection id, used verbatim instead of resolving `satellite`
        through `source.collection`. Defaults to None, which resolves `satellite` as
        usual - needed for a satellite with no `gfetch.sources` registry entry.
    intersects : dict | None
        GeoJSON-like geometry mapping (e.g. `shapely.geometry.mapping(polygon)`),
        used instead of `bbox` for the actual STAC query - the STAC API spec treats
        `bbox`/`intersects` as mutually exclusive, so `bbox` is dropped from the
        request when this is given (a `gfetch.countries`-derived AOI passes its
        exact country polygon here rather than just its bounding box). Defaults to
        None (search by `bbox`).

    Returns
    -------
    list[pystac.Item]
        Matching STAC items.
    """
    collection = collection or source.collection(satellite)
    try:
        collection = resolve_collection(source, collection, *_parse_date_range(datetime))
    except ValueError as e:
        log.error(e)
        raise
    log.info(
        f"Searching {source.name} collection {collection!r} for bbox={bbox} datetime={datetime}"
    )
    log.debug(f"Opening STAC catalog at {source.api_url}")
    catalog = pystac_client.Client.open(source.api_url)
    result = catalog.search(
        collections=[collection],
        bbox=None if intersects is not None else bbox,
        intersects=intersects,
        datetime=datetime,
        query=query,
        max_items=max_items,
    )

    matched = result.matched()
    log.info(f"{matched} item(s) match; fetching..." if matched is not None else "Fetching...")

    items: list[pystac.Item] = []
    for page_num, page in enumerate(result.pages(), start=1):
        items.extend(page.items)
        total = f"/{matched}" if matched is not None else ""
        log.debug(f"Fetched page {page_num} ({len(items)}{total} items so far)")
    if satellite == "sentinel-2":
        deduped = _dedupe_sentinel2_processing_baseline(items)
        if len(deduped) != len(items):
            log.info(
                f"Dropped {len(items) - len(deduped)} superseded-processing-baseline duplicate(s)"
            )
        items = deduped
    log.info(f"Found {len(items)} items")
    return items
