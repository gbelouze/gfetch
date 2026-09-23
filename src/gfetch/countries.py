"""Country-boundary AOI resolution: derive a bounding box (and, for the `search`
stage, an exact polygon for server-side filtering) from a list of country names,
instead of requiring an explicit AOI bounding box. Mirrors geefetch's `aoi.country`
config field (`geefetch/cli/download_implementation.py::load_country_filter_polygon`),
against the same public boundaries dataset - geefetch's own use of the polygon is
coarse (tile-level `.intersects()`, never a per-pixel clip; never reaches the GEDI
vector path at all), so matching it here means resolving a bounding box, not clipping
downloaded data to the exact country shape.
"""

import logging
import os
from collections.abc import Sequence
from difflib import get_close_matches
from pathlib import Path

import geopandas as gpd
import requests
import shapely

log = logging.getLogger(__name__)

__all__ = ["resolve_country_polygon"]

COUNTRY_BORDERS_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "world-administrative-boundaries/exports/geojson"
)

# XDG_CACHE_HOME, not gfetch's own per-job output_dir/cache: this is a small, global,
# cross-job resource (country boundaries), unrelated to any one job's asset cache.
_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "gfetch"
_CACHE_PATH = _CACHE_DIR / "world-administrative-boundaries.geojson"


def _country_borders(cache_path: Path = _CACHE_PATH) -> gpd.GeoDataFrame:
    """
    Load the world country-boundaries dataset, downloading and caching it on first
    use.

    Parameters
    ----------
    cache_path : Path
        Local path to cache the downloaded GeoJSON at. Defaults to `_CACHE_PATH`
        (under `XDG_CACHE_HOME`, or `~/.cache` if unset).

    Returns
    -------
    geopandas.GeoDataFrame
        One row per country, with a `name` column and its boundary `geometry`.
    """
    if not cache_path.exists():
        log.info(f"Downloading country boundaries to {cache_path}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(COUNTRY_BORDERS_URL, timeout=60)
        response.raise_for_status()
        cache_path.write_bytes(response.content)
    return gpd.read_file(cache_path)


def _suggest(name: str, known_names: list[str]) -> str:
    """
    Build a "did you mean" hint for an unrecognized country name.

    Parameters
    ----------
    name : str
        The name that didn't match.
    known_names : list[str]
        Every valid country name in the boundaries dataset.

    Returns
    -------
    str
        A `" Did you mean 'X'?"` hint, or `""` if nothing close was found. A
        case-insensitive substring match is tried first, since official dataset
        names often differ a lot in length from common short names (e.g.
        'Tanzania' -> 'United Republic of Tanzania') - too big a length gap for
        `difflib`'s ratio-based matching alone to reliably catch.
    """
    substring_hits = [n for n in known_names if name.lower() in n.lower()]
    if len(substring_hits) == 1:
        return f" Did you mean {substring_hits[0]!r}?"
    close = get_close_matches(name, known_names, n=1, cutoff=0.3)
    return f" Did you mean {close[0]!r}?" if close else ""


def resolve_country_polygon(countries: Sequence[str]) -> shapely.geometry.base.BaseGeometry:
    """
    Resolve the union of one or more countries' boundaries.

    Parameters
    ----------
    countries : Sequence[str]
        Country names, matched exactly (case-sensitive) against the `name` column
        of the `world-administrative-boundaries` dataset.

    Returns
    -------
    shapely.geometry.base.BaseGeometry
        The union of `countries`' boundary geometries (a `Polygon`/`MultiPolygon`
        in practice), in EPSG:4326.

    Raises
    ------
    ValueError
        If a country name doesn't match any entry in the dataset. The message
        suggests the closest known name, if any.
    """
    borders = _country_borders()
    all_names = borders.name.tolist()
    polygons = []
    for name in countries:
        match = borders[borders.name == name]
        if match.empty:
            hint = _suggest(name, all_names)
            raise ValueError(f"Unknown country {name!r}.{hint}")
        polygons.append(match.iloc[0].geometry)
    log.info(
        f"Resolved {len(countries)} countr{'y' if len(countries) == 1 else 'ies'}: {countries}"
    )
    return shapely.union_all(polygons)
