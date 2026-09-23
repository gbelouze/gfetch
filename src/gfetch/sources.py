"""STAC source registry: maps a source name to its API endpoint and per-satellite
collection ids.

`stac-asset` auto-selects the right download client (plain HTTP, S3, Planetary
Computer SAS-signing, ...) per asset href, so a source here needs nothing beyond its
search endpoint and collection-id mapping.
"""

import datetime
from dataclasses import dataclass

__all__ = ["SOURCES", "StacSource", "check_collection_coverage", "get_source"]

_SENTINEL2_COLLECTIONS = {
    "earthsearch": "sentinel-2-c1-l2a",
    "planetary-computer": "sentinel-2-l2a",
}

_SENTINEL1_COLLECTIONS = {
    "earthsearch": "sentinel-1-grd",
    "planetary-computer": "sentinel-1-grd",
}

# Periods a collection is known to be missing, as (first day, last day) inclusive, keyed
# by (source name, collection id). Earth Search's Collection 1 only holds data
# reprocessed by ESA to baseline 05.00+, and per Earth Search's own documentation (as of
# April 2024) that reprocessing hasn't reached Nov 2016 - Nov 2019 or 2022. Confirmed
# live 2026-09-23: 45 items for Jan-Sep 2022 over a 230 km box in France, against 1,372
# in `sentinel-2-l2a`.
_COLLECTION_GAPS: dict[tuple[str, str], list[tuple[datetime.date, datetime.date]]] = {
    ("earthsearch", "sentinel-2-c1-l2a"): [
        (datetime.date(2016, 11, 1), datetime.date(2019, 11, 30)),
        (datetime.date(2022, 1, 1), datetime.date(2022, 12, 31)),
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


def check_collection_coverage(
    source: StacSource,
    collection: str,
    start: datetime.date | None,
    end: datetime.date | None,
) -> None:
    """
    Check a time range against the periods a collection is known to be missing.

    Parameters
    ----------
    source : StacSource
        STAC API source the collection is searched on.
    collection : str
        Collection id.
    start : datetime.date | None
        First day of the time range, or None if open-ended.
    end : datetime.date | None
        Last day of the time range, or None if open-ended.

    Raises
    ------
    ValueError
        If the time range overlaps a known gap of `collection` on `source`.
    """
    for gap_start, gap_end in _COLLECTION_GAPS.get((source.name, collection), []):
        if (start is None or start <= gap_end) and (end is None or end >= gap_start):
            raise ValueError(
                f"{source.name} collection {collection!r} is missing data from {gap_start} "
                f"to {gap_end}, which overlaps the requested time range {start}/{end}. "
                "Choose a time range outside that period, or another source."
            )
