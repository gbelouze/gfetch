"""STAC source registry: maps a source name to its API endpoint and per-satellite
collection ids.

`stac-asset` auto-selects the right download client (plain HTTP, S3, Planetary
Computer SAS-signing, ...) per asset href, so a source here needs nothing beyond its
search endpoint and collection-id mapping.
"""

import datetime
import logging
from dataclasses import dataclass

__all__ = ["SOURCES", "StacSource", "get_source", "resolve_collection"]

log = logging.getLogger(__name__)

_SENTINEL2_COLLECTIONS = {
    "earthsearch": "sentinel-2-c1-l2a",
    "planetary-computer": "sentinel-2-l2a",
}

_SENTINEL1_COLLECTIONS = {
    "earthsearch": "sentinel-1-grd",
    "planetary-computer": "sentinel-1-grd",
}


@dataclass(frozen=True)
class _CollectionGap:
    """
    A period a collection is known to be missing.

    Attributes
    ----------
    start : datetime.date
        First missing day.
    end : datetime.date
        Last missing day.
    fallback : str | None
        Collection on the same source to search instead when a time range falls
        entirely within the gap, or None to refuse such time ranges.
    fallback_caveat : str
        Appended to the warning logged when `fallback` is used. Defaults to ''.
    """

    start: datetime.date
    end: datetime.date
    fallback: str | None = None
    fallback_caveat: str = ""


# Keyed by (source name, collection id). Earth Search's Collection 1 only holds data
# reprocessed by ESA to baseline 05.00+, and per Earth Search's own documentation (as of
# April 2024) that reprocessing hasn't reached Nov 2016 - Nov 2019 or 2022. Confirmed
# live 2026-09-23: 45 items for Jan-Sep 2022 over a 230 km box in France, against 1,372
# in `sentinel-2-l2a`. That older collection has the same asset keys, `grid:code` and
# `raster:bands` scale/offset as Collection 1, but its items before 25 Jan 2022 are
# baseline 03.01, whose DNs lack the +1000 offset every later baseline carries.
_COLLECTION_GAPS: dict[tuple[str, str], list[_CollectionGap]] = {
    ("earthsearch", "sentinel-2-c1-l2a"): [
        _CollectionGap(datetime.date(2016, 11, 1), datetime.date(2019, 11, 30)),
        _CollectionGap(
            datetime.date(2022, 1, 1),
            datetime.date(2022, 12, 31),
            fallback="sentinel-2-l2a",
            fallback_caveat=(
                "Its items before 2022-01-25 are processing baseline 03.01, whose "
                "reflectance DNs lack the +1000 offset of every later baseline and of "
                "Collection 1; gfetch does not harmonize them."
            ),
        ),
    ],
}

_COLLECTIONS: dict[str, dict[str, str]] = {
    "sentinel-2": _SENTINEL2_COLLECTIONS,
    "sentinel-1": _SENTINEL1_COLLECTIONS,
}


@dataclass(frozen=True)
class StacSource:
    """
    A STAC API source: an endpoint plus its per-satellite collection ids.

    Attributes
    ----------
    name : str
        Short identifier for the source (e.g. 'earthsearch', 'planetary-computer').
    api_url : str
        Root URL of the STAC API to search against.
    """

    name: str
    api_url: str

    def collection(self, satellite: str) -> str:
        """
        Resolve the collection id this source uses for a given satellite.

        Parameters
        ----------
        satellite : str
            Satellite/profile name (e.g. 'sentinel-2').

        Returns
        -------
        str
            The collection id to search against on this source.
        """
        try:
            return _COLLECTIONS[satellite][self.name]
        except KeyError as e:
            raise ValueError(
                f"No known collection for satellite {satellite!r} on source {self.name!r}"
            ) from e


SOURCES: dict[str, StacSource] = {
    "earthsearch": StacSource("earthsearch", "https://earth-search.aws.element84.com/v1"),
    "planetary-computer": StacSource(
        "planetary-computer", "https://planetarycomputer.microsoft.com/api/stac/v1"
    ),
}


def get_source(name: str) -> StacSource:
    """
    Look up a registered STAC source by name.

    Parameters
    ----------
    name : str
        Source name, one of the keys in `SOURCES`.

    Returns
    -------
    StacSource
        The corresponding source.
    """
    try:
        return SOURCES[name]
    except KeyError as e:
        raise ValueError(f"Unknown source {name!r}; known sources: {sorted(SOURCES)}") from e


def resolve_collection(
    source: StacSource,
    collection: str,
    start: datetime.date | None,
    end: datetime.date | None,
) -> str:
    """
    Resolve the collection to search for a time range, given the periods a collection
    is known to be missing.

    Parameters
    ----------
    source : StacSource
        STAC API source the collection is searched on.
    collection : str
        Requested collection id.
    start : datetime.date | None
        First day of the time range, or None if open-ended.
    end : datetime.date | None
        Last day of the time range, or None if open-ended.

    Returns
    -------
    str
        `collection` if the time range overlaps none of its known gaps, else the
        fallback collection of the gap the time range falls entirely within.

    Raises
    ------
    ValueError
        If the time range overlaps a known gap of `collection` on `source` that has no
        fallback, or that it doesn't fall entirely within.
    """
    for gap in _COLLECTION_GAPS.get((source.name, collection), []):
        if not ((start is None or start <= gap.end) and (end is None or end >= gap.start)):
            continue
        within = start is not None and end is not None and gap.start <= start and end <= gap.end
        if gap.fallback is not None and within:
            log.warning(
                f"!!! {source.name} collection {collection!r} is missing data from "
                f"{gap.start} to {gap.end}: searching {gap.fallback!r} instead for "
                f"{start}/{end}. Its items are not Collection 1 reprocessed data, so "
                f"this mosaic is not radiometrically equivalent to one outside that "
                f"period. {gap.fallback_caveat}".rstrip()
            )
            return gap.fallback
        hint = (
            f"Split the time range so that each part lies either entirely outside or "
            f"entirely within that period ({gap.fallback!r} is searched instead within it)."
            if gap.fallback is not None
            else "Choose a time range outside that period, or another source."
        )
        raise ValueError(
            f"{source.name} collection {collection!r} is missing data from {gap.start} "
            f"to {gap.end}, which overlaps the requested time range {start}/{end}. {hint}"
        )
    return collection
