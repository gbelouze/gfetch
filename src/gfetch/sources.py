"""STAC source registry: maps a source name to its API endpoint and per-satellite
collection ids.

`stac-asset` auto-selects the right download client (plain HTTP, S3, Planetary
Computer SAS-signing, ...) per asset href, so a source here needs nothing beyond its
search endpoint and collection-id mapping.
"""

from dataclasses import dataclass

__all__ = ["SOURCES", "StacSource", "get_source"]

_SENTINEL2_COLLECTIONS = {
    "earthsearch": "sentinel-2-c1-l2a",
    "planetary-computer": "sentinel-2-l2a",
}

_SENTINEL1_COLLECTIONS = {
    "earthsearch": "sentinel-1-grd",
    "planetary-computer": "sentinel-1-grd",
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
