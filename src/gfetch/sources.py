"""STAC source registry: maps a source name to its API endpoint and per-satellite
collection ids.

`stac-asset` auto-selects the right download client (plain HTTP, S3, ...) per asset
href, so a source here needs nothing beyond its search endpoint and collection-id
mapping. Planetary Computer hrefs need signing, by `planetary_computer_signer`, both to
download them and to read them straight from their hrefs.
"""

import datetime
import json
import logging
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

from retry import retry

__all__ = [
    "SOURCES",
    "HeterogeneousFallbackError",
    "StacSource",
    "asset_renames",
    "get_source",
    "planetary_computer_signer",
    "resolve_collection",
]

log = logging.getLogger(__name__)

_PC_SAS_TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
_AZURE_BLOB_DOMAIN = ".blob.core.windows.net"
_PC_TOKEN_REFRESH_MARGIN = datetime.timedelta(minutes=5)

_SENTINEL2_COLLECTIONS = {
    "earthsearch": "sentinel-2-c1-l2a",
    "planetary-computer": "sentinel-2-l2a",
}

# Earth Search only has GRD, which isn't terrain-corrected: its geolocation is off by
# up to several hundred meters over relief. Planetary Computer's RTC is, and it reads
# anonymously once signed despite its collection's `msft:requires_account: true`.
_SENTINEL1_COLLECTIONS = {
    "earthsearch": "sentinel-1-grd",
    "planetary-computer": "sentinel-1-rtc",
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
    confirm : bool
        Whether using `fallback` must be explicitly accepted, see
        `HeterogeneousFallbackError`. Defaults to False.
    """

    start: datetime.date
    end: datetime.date
    fallback: str | None = None
    fallback_caveat: str = ""
    confirm: bool = False


class HeterogeneousFallbackError(ValueError):
    """
    A time range lies within a gap whose fallback collection must be explicitly accepted.

    Raised by `resolve_collection` unless called with `accept_heterogeneous=True`.
    """


# Keyed by (source name, collection id). Earth Search's Collection 1 only holds data
# reprocessed by ESA to baseline 05.00+. Measured live 2026-10-16 over France, Mozambique,
# Japan and the US (`claude/tech-stack.md`): it holds nothing before Nov 2017, a fraction
# of the scenes until 21 Mar 2019, and none between 1 Jan and 5 Dec 2022. The older
# `sentinel-2-l2a` has the same asset keys and `grid:code` as Collection 1. Its DNs
# lack Collection 1's +1000 offset, except most items acquired 25 Jan - 27 Feb 2022
# (`claude/s2-2022/README.md`). Neither `earthsearch:boa_offset_applied` nor
# `raster:bands` offset tells which, see `gfetch.dn`. All of this is specific to Earth
# Search v1: revisit it once v2 leaves beta (`claude/s2-dn-harmonization.md`).
_COLLECTION_GAPS: dict[tuple[str, str], list[_CollectionGap]] = {
    ("earthsearch", "sentinel-2-c1-l2a"): [
        _CollectionGap(
            datetime.date(2015, 6, 23),
            datetime.date(2019, 3, 21),
            fallback="sentinel-2-l2a",
            fallback_caveat=(
                "Its items mix processing baselines 00.01 to 03.01 (older Sen2Cor "
                "versions, likely without geometric refinement before 03.00), in one "
                "composite and against Collection 1's 05.00+; gfetch only harmonizes "
                "their DN offset."
            ),
            confirm=True,
        ),
        _CollectionGap(
            datetime.date(2022, 1, 1),
            datetime.date(2022, 12, 5),
            fallback="sentinel-2-l2a",
            fallback_caveat=(
                "Its reflectance DNs lack Collection 1's +1000 offset, except most "
                "items acquired 2022-01-25 to 2022-02-27; gfetch harmonizes the offset "
                "per item, see `gfetch.dn`."
            ),
        ),
    ],
}

_COLLECTIONS: dict[str, dict[str, str]] = {
    "sentinel-2": _SENTINEL2_COLLECTIONS,
    "sentinel-1": _SENTINEL1_COLLECTIONS,
}

# Keyed by (source name, collection id): asset key -> the key gfetch uses, so a config
# names bands the same on every source. Planetary Computer keys Sentinel-2 assets by band
# id, Earth Search by the names below, which match its own `eo:bands` metadata. They can't
# be derived from Planetary Computer's `eo:bands`: it gives B05-B07 and B8A the same
# `rededge` common name, and B09 none.
_ASSET_RENAMES: dict[tuple[str, str], dict[str, str]] = {
    ("planetary-computer", "sentinel-2-l2a"): {
        "B01": "coastal",
        "B02": "blue",
        "B03": "green",
        "B04": "red",
        "B05": "rededge1",
        "B06": "rededge2",
        "B07": "rededge3",
        "B08": "nir",
        "B8A": "nir08",
        "B09": "nir09",
        "B11": "swir16",
        "B12": "swir22",
        "SCL": "scl",
        "AOT": "aot",
        "WVP": "wvp",
    },
}


def asset_renames(source: "StacSource", collection: str) -> dict[str, str]:
    """
    Look up how a collection's asset keys are renamed into the keys gfetch uses.

    Parameters
    ----------
    source : StacSource
        STAC API source the collection is searched on.
    collection : str
        Collection id.

    Returns
    -------
    dict[str, str]
        Asset key -> gfetch key, empty if the collection's keys are used as they are.
    """
    return _ASSET_RENAMES.get((source.name, collection), {})


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
    *,
    accept_heterogeneous: bool = False,
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
    accept_heterogeneous : bool
        Use the fallback of a gap that requires confirmation instead of raising.
        Defaults to False.

    Returns
    -------
    str
        `collection` if the time range overlaps none of its known gaps, else the
        fallback collection of the gap the time range falls entirely within.

    Raises
    ------
    HeterogeneousFallbackError
        If the time range falls entirely within a gap whose fallback requires
        confirmation, and `accept_heterogeneous` is False.
    ValueError
        If the time range overlaps a known gap of `collection` on `source` that has no
        fallback, or that it doesn't fall entirely within.
    """
    for gap in _COLLECTION_GAPS.get((source.name, collection), []):
        if not ((start is None or start <= gap.end) and (end is None or end >= gap.start)):
            continue
        within = start is not None and end is not None and gap.start <= start and end <= gap.end
        if gap.fallback is not None and within:
            message = (
                f"!!! {source.name} collection {collection!r} is missing data from "
                f"{gap.start} to {gap.end}: searching {gap.fallback!r} instead for "
                f"{start}/{end}. Its items are not Collection 1 reprocessed data, so "
                f"this mosaic is not radiometrically equivalent to one outside that "
                f"period. {gap.fallback_caveat}".rstrip()
            )
            if gap.confirm and not accept_heterogeneous:
                raise HeterogeneousFallbackError(message)
            log.warning(message)
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


# Planetary Computer rate-limits anonymous token requests (HTTP 429). Waits 1, 2, 4, ...,
# 32, then 60 s between attempts: about 5 minutes in total.
@retry(urllib.error.URLError, tries=10, delay=1, backoff=2, max_delay=60, logger=log)
def _pc_sas_token(account: str, container: str) -> tuple[str, datetime.datetime]:
    """
    Request a new, anonymous SAS token for a Planetary Computer blob container.

    Parameters
    ----------
    account : str
        Azure storage account name.
    container : str
        Blob container name.

    Returns
    -------
    tuple[str, datetime.datetime]
        The token, as a URL query string, and when it expires.
    """
    url = f"{_PC_SAS_TOKEN_URL}/{account}/{container}"
    with urllib.request.urlopen(url, timeout=30) as response:
        data = json.load(response)
    return data["token"], datetime.datetime.fromisoformat(data["msft:expiry"])


def planetary_computer_signer() -> Callable[[str], str]:
    """
    Build a function appending a SAS token to Planetary Computer asset hrefs.

    Planetary Computer's STAC items hold unsigned Azure Blob Storage hrefs, which
    Azure refuses (HTTP 409) without a token. Each signer requests a token the first
    time it meets a container, then reuses it until it is within
    `_PC_TOKEN_REFRESH_MARGIN` of expiring (tokens last about 45 minutes), so one
    signer serves a run of any length. It is thread-safe, and concurrent calls for
    the same container share a single token request. Hrefs outside Azure Blob
    Storage, or already holding a query string, are returned unchanged.

    Returns
    -------
    Callable[[str], str]
        Maps an href to its signed version, e.g. as `odc.stac.load`'s `patch_url`.
    """
    tokens: dict[tuple[str, str], tuple[str, datetime.datetime]] = {}
    lock = threading.Lock()

    def sign(href: str) -> str:
        url = urlparse(href)
        if not url.netloc.endswith(_AZURE_BLOB_DOMAIN) or url.query:
            return href
        account = url.netloc.removesuffix(_AZURE_BLOB_DOMAIN)
        container = url.path.lstrip("/").split("/", 1)[0]
        with lock:
            cached = tokens.get((account, container))
            now = datetime.datetime.now(datetime.UTC)
            if cached is None or cached[1] - now < _PC_TOKEN_REFRESH_MARGIN:
                cached = tokens[account, container] = _pc_sas_token(account, container)
        return f"{href}?{cached[0]}"

    return sign
