"""GEDI vector fetch via SlideRule: on-demand, server-side subsetting of remote GEDI
HDF5 granules (https://slideruleearth.io). Unlike gfetch's raster pipeline, this needs
no separate search/download stage: SlideRule resolves the matching granules via CMR and
performs the byte-range subsetting itself, so one request returns an already-subsetted
GeoDataFrame of footprints. No credentials are required against SlideRule's public
cluster (the default) - NASA Earthdata authentication against the source DAAC is handled
server-side; it's only needed for a self-hosted SlideRule deployment.
"""

import hashlib
import json
import logging
import math
import os
import tempfile
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely
from pyproj import Geod
from rich.progress import Progress
from shapely.geometry.base import BaseGeometry
from sliderule import gedi, sliderule
from sliderule.session import Session

from gfetch.utils.geoparquet import write_geoparquet
from gfetch.utils.progress import temporary_task

log = logging.getLogger(__name__)

# sliderule.init() hardcodes its requests.Session's trust_env=False with no way to
# override it, so it silently ignores http(s)_proxy/no_proxy - unusable on a network
# that requires an explicit proxy (e.g. an HPC cluster whose compute nodes have no
# direct internet route). Assigning a correctly-configured Session to the module
# global ourselves (sliderule.slideruleSession = sliderule.create_session(trust_env=
# True)) turned out to not reliably reach every internal call path (gedi02ap() still
# ended up with a trust_env=False session even from a freshly-started process, root
# cause unconfirmed) - patching the constructor's own default instead, so every
# Session built anywhere in the process (including SlideRule's own lazy re-init
# inside checksession()) gets trust_env=True unless a caller explicitly overrides it.
_orig_session_init = Session.__init__


def _patched_session_init(self: Session, *args: object, **kwargs: object) -> None:
    kwargs.setdefault("trust_env", True)
    _orig_session_init(self, *args, **kwargs)


Session.__init__ = _patched_session_init

__all__ = [
    "GEDI_L2A_DEFAULT_FIELDS",
    "GEDI_L2A_RH_PERCENTILES",
    "GEDI_L4A_DEFAULT_FIELDS",
    "GEDI_MAX_TILE_SIZE_M",
    "expand_rh",
    "fetch_gedi_l2a",
    "fetch_gedi_l4a",
    "split_bbox",
]

# gedi02ap already returns a fixed, reduced set of L2A fields (not the full ~100
# variable granule); elevation_lm/elevation_hr are the two headline
# ground-elevation/canopy-top measurements. The full set SlideRule exposes is
# ('orbit', 'solar_elevation', 'track', 'elevation_lm', 'elevation_hr', 'sensitivity',
# 'flags', 'beam'). Additional per-shot fields (e.g. 'rh', 'quality_flag') can be
# read straight out of the source granule via `anc_fields`, bypassing this ceiling.
GEDI_L2A_DEFAULT_FIELDS: tuple[str, ...] = ("elevation_lm", "elevation_hr")

# GEDI L2A's `rh` ancillary field is a 101-element per-shot array where index i is
# the i-th relative-height percentile; these eight are the ones geefetch's GEE-backed
# GEDI L2A image collection exposed as named bands.
GEDI_L2A_RH_PERCENTILES: tuple[int, ...] = (0, 2, 25, 50, 75, 95, 98, 100)

# geefetch's L2A quality filter, minus its full-power-beam condition (commented out
# there too). `quality_flag == 1` and `degrade_flag == 0` are applied server-side via
# SlideRule's `l2_quality_filter`/`degrade_filter`; the rest have no SlideRule
# equivalent and are applied to each tile's response.
GEDI_L2A_MAX_SOLAR_ELEVATION: float = 0.0
GEDI_L2A_MIN_SENSITIVITY: float = 0.9
GEDI_L2A_RH98_RANGE: tuple[float, float] = (0.0, 80.0)

# gedi04ap's fixed schema is ('agbd', 'elevation', 'sensitivity', 'solar_elevation',
# 'orbit', 'track', 'beam', 'flags'); agbd (aboveground biomass density, Mg/ha) is
# L4A's headline measurement.
GEDI_L4A_DEFAULT_FIELDS: tuple[str, ...] = ("agbd", "elevation")

# geefetch's L4A quality filter. `l4_quality_flag == 1` and `degrade_flag == 0` are
# applied server-side via SlideRule's `l4_quality_filter`/`degrade_filter`, the
# sensitivity threshold to each tile's response.
GEDI_L4A_MIN_SENSITIVITY: float = 0.9

# A single SlideRule request over a large AOI is unreliable (server-side failures
# surfacing as an empty response), so each product is fetched with one request per
# tile of at most this size on each side.
GEDI_MAX_TILE_SIZE_M: float = 50_000.0

_GEOD = Geod(ellps="WGS84")


def _bbox_to_poly(bbox: tuple[float, float, float, float]) -> list[dict[str, float]]:
    """
    Convert a bounding box to SlideRule's polygon format: a closed, counter-clockwise
    ring of `{"lon": ..., "lat": ...}` points.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Bounding box (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.

    Returns
    -------
    list[dict[str, float]]
        Polygon ring, first and last point equal, as required by SlideRule's `poly`
        request parameter.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    return [
        {"lon": min_lon, "lat": min_lat},
        {"lon": max_lon, "lat": min_lat},
        {"lon": max_lon, "lat": max_lat},
        {"lon": min_lon, "lat": max_lat},
        {"lon": min_lon, "lat": min_lat},
    ]


def split_bbox(
    bbox: tuple[float, float, float, float], max_size_m: float = GEDI_MAX_TILE_SIZE_M
) -> list[tuple[float, float, float, float]]:
    """
    Split a bounding box into an even grid of tiles at most `max_size_m` on each side.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Bounding box (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
    max_size_m : float
        Maximum tile width and height, in meters on the WGS84 ellipsoid. Defaults to
        `GEDI_MAX_TILE_SIZE_M`.

    Returns
    -------
    list[tuple[float, float, float, float]]
        Tiles covering `bbox` exactly, row-major from the south-west corner, sharing
        their edges with their neighbours.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    # A degree of longitude is widest at the latitude closest to the equator.
    widest_lat = 0.0 if min_lat <= 0.0 <= max_lat else min(min_lat, max_lat, key=abs)
    width_m = _GEOD.line_length([min_lon, max_lon], [widest_lat, widest_lat])
    height_m = _GEOD.line_length([min_lon, min_lon], [min_lat, max_lat])
    n_x = max(1, math.ceil(width_m / max_size_m))
    n_y = max(1, math.ceil(height_m / max_size_m))
    lons = [min_lon + (max_lon - min_lon) * i / n_x for i in range(n_x)] + [max_lon]
    lats = [min_lat + (max_lat - min_lat) * j / n_y for j in range(n_y)] + [max_lat]
    return [(lons[i], lats[j], lons[i + 1], lats[j + 1]) for j in range(n_y) for i in range(n_x)]


def _base_parms(
    tile: tuple[float, float, float, float],
    time_range: tuple[str, str] | None,
    anc_fields: Sequence[str],
) -> dict:
    """
    Build the SlideRule request parameters shared by every GEDI product.

    Parameters
    ----------
    tile : tuple[float, float, float, float]
        Tile (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
    time_range : tuple[str, str] | None
        See `fetch_gedi_l2a`.
    anc_fields : Sequence[str]
        Ancillary fields to request, possibly empty.

    Returns
    -------
    dict
        `parms` for `gedi02ap`/`gedi04ap`.
    """
    parms: dict = {"poly": _bbox_to_poly(tile)}
    if time_range is not None:
        parms["t0"], parms["t1"] = time_range
    if anc_fields:
        parms["anc_fields"] = list(anc_fields)
    return parms


def _request_tile(
    endpoint: str,
    parms: dict,
    tile: tuple[float, float, float, float],
    bbox: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    """
    Send one SlideRule GEDI request over a tile and drop footprints owned by a
    neighbouring tile.

    Parameters
    ----------
    endpoint : str
        `sliderule.gedi` function to call, e.g. `'gedi02ap'`.
    parms : dict
        Request parameters, see `_base_parms`.
    tile : tuple[float, float, float, float]
        Tile (min_lon, min_lat, max_lon, max_lat) in EPSG:4326, as returned by
        `split_bbox(bbox)`.
    bbox : tuple[float, float, float, float]
        The full AOI `tile` was split from.

    Returns
    -------
    geopandas.GeoDataFrame
        Footprints in `tile`, excluding those on its north/east edges unless those
        edges are also `bbox`'s, so that no footprint is returned by two tiles.
    """
    log.debug(f"Requesting {endpoint} footprints over tile={tile}, parms={parms}")
    gdf = getattr(gedi, endpoint)(parms)
    if not gdf.empty:
        _, _, max_lon, max_lat = tile
        x, y = gdf.geometry.x, gdf.geometry.y
        keep_x = x <= max_lon if max_lon == bbox[2] else x < max_lon
        keep_y = y <= max_lat if max_lat == bbox[3] else y < max_lat
        gdf = gdf[keep_x & keep_y]
    return gdf


def _select_columns(
    gdf: gpd.GeoDataFrame, keep: Sequence[str], tile: tuple[float, float, float, float]
) -> gpd.GeoDataFrame:
    """
    Keep only the given columns (plus geometry) of one tile's response.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        One tile's footprints.
    keep : Sequence[str]
        Columns to keep, in order.
    tile : tuple[float, float, float, float]
        The tile `gdf` covers, for logging.

    Returns
    -------
    gpd.GeoDataFrame
        `gdf` restricted to `keep`. A column missing from an empty response is
        added empty.

    Raises
    ------
    KeyError
        If a column of `keep` is missing from a non-empty response.
    """
    missing = [f for f in keep if f not in gdf.columns]
    if missing:
        if not gdf.empty:
            raise KeyError(f"field(s) {missing} not in response columns {list(gdf.columns)}")
        # SlideRule returns a bare-geometry frame with no data columns at all
        # (`sliderule.emptyframe()`) both when the AOI/time range genuinely matches
        # zero footprints and when the request itself failed server-side (SlideRule
        # logs that failure loudly on its own, e.g. "Unexpected termination of
        # response") - either way, this column selection would otherwise raise a
        # confusing KeyError instead of a well-formed empty result.
        log.warning(
            f"Empty response over tile={tile} with no data columns (missing {missing}) - "
            "the request may have failed server-side rather than genuinely matching zero "
            "footprints"
        )
        for f in missing:
            gdf[f] = []
    return gdf[[*keep, gdf.geometry.name]]


def _fetch_l2a_tile(
    tile: tuple[float, float, float, float],
    bbox: tuple[float, float, float, float],
    time_range: tuple[str, str] | None,
    fields: Sequence[str] | None,
    anc_fields: Sequence[str] | None,
    rh_percentiles: Sequence[int] | None,
    quality_filter: bool,
) -> gpd.GeoDataFrame:
    """
    Fetch GEDI L2A footprints over one tile of a larger AOI.

    Parameters
    ----------
    tile : tuple[float, float, float, float]
        Tile (min_lon, min_lat, max_lon, max_lat) in EPSG:4326, as returned by
        `split_bbox(bbox)`.
    bbox : tuple[float, float, float, float]
        The full AOI `tile` was split from.
    time_range : tuple[str, str] | None
        See `fetch_gedi_l2a`.
    fields : Sequence[str] | None
        See `fetch_gedi_l2a`.
    anc_fields : Sequence[str] | None
        See `fetch_gedi_l2a`.
    rh_percentiles : Sequence[int] | None
        See `fetch_gedi_l2a`.
    quality_filter : bool
        See `fetch_gedi_l2a`.

    Returns
    -------
    geopandas.GeoDataFrame
        Footprints in `tile` (see `_request_tile`).
    """
    need_rh = bool(rh_percentiles) or quality_filter
    request_anc_fields = list(dict.fromkeys([*(anc_fields or ()), *(["rh"] if need_rh else [])]))
    parms = _base_parms(tile, time_range, request_anc_fields)
    if quality_filter:
        parms["l2_quality_filter"] = True
        parms["degrade_filter"] = True
    gdf = _request_tile("gedi02ap", parms, tile, bbox)

    if quality_filter and not gdf.empty:
        rh98 = gdf["rh"].map(lambda arr: arr[98])
        gdf = gdf[
            (gdf["solar_elevation"] <= GEDI_L2A_MAX_SOLAR_ELEVATION)
            & (gdf["sensitivity"] >= GEDI_L2A_MIN_SENSITIVITY)
            & rh98.between(*GEDI_L2A_RH98_RANGE)
        ]

    rh_columns = [f"rh{p}" for p in rh_percentiles or ()]
    if "rh" in gdf.columns:
        if rh_percentiles:
            gdf = expand_rh(gdf, rh_percentiles)
        elif "rh" not in (anc_fields or ()):
            gdf = gdf.drop(columns="rh")

    if fields is None:
        return gdf
    kept_anc_fields = [f for f in anc_fields or () if not (f == "rh" and rh_percentiles)]
    return _select_columns(gdf, list(dict.fromkeys([*fields, *kept_anc_fields, *rh_columns])), tile)


def _fetch_l4a_tile(
    tile: tuple[float, float, float, float],
    bbox: tuple[float, float, float, float],
    time_range: tuple[str, str] | None,
    fields: Sequence[str] | None,
    anc_fields: Sequence[str] | None,
    quality_filter: bool,
) -> gpd.GeoDataFrame:
    """
    Fetch GEDI L4A footprints over one tile of a larger AOI.

    Parameters
    ----------
    tile : tuple[float, float, float, float]
        Tile (min_lon, min_lat, max_lon, max_lat) in EPSG:4326, as returned by
        `split_bbox(bbox)`.
    bbox : tuple[float, float, float, float]
        The full AOI `tile` was split from.
    time_range : tuple[str, str] | None
        See `fetch_gedi_l4a`.
    fields : Sequence[str] | None
        See `fetch_gedi_l4a`.
    anc_fields : Sequence[str] | None
        See `fetch_gedi_l4a`.
    quality_filter : bool
        See `fetch_gedi_l4a`.

    Returns
    -------
    geopandas.GeoDataFrame
        Footprints in `tile` (see `_request_tile`).
    """
    parms = _base_parms(tile, time_range, anc_fields or ())
    if quality_filter:
        parms["l4_quality_filter"] = True
        parms["degrade_filter"] = True
    gdf = _request_tile("gedi04ap", parms, tile, bbox)

    if quality_filter and not gdf.empty:
        gdf = gdf[gdf["sensitivity"] >= GEDI_L4A_MIN_SENSITIVITY]

    if fields is None:
        return gdf
    return _select_columns(gdf, list(dict.fromkeys([*fields, *(anc_fields or ())])), tile)


def _fetch_tiles(
    product: str,
    fetch_tile: Callable[[tuple[float, float, float, float]], gpd.GeoDataFrame],
    bbox: tuple[float, float, float, float],
    max_size_m: float,
    polygon: BaseGeometry | None,
    tile_dir: Path | None,
    params: dict,
    progress: Progress | None,
) -> gpd.GeoDataFrame:
    """
    Fetch one GEDI product tile by tile over an AOI, resuming from saved tiles.

    Parameters
    ----------
    product : str
        Product name for logging, e.g. `'L2A'`.
    fetch_tile : Callable[[tuple[float, float, float, float]], gpd.GeoDataFrame]
        Fetches one tile's footprints.
    bbox : tuple[float, float, float, float]
        See `fetch_gedi_l2a`.
    max_size_m : float
        See `fetch_gedi_l2a`.
    polygon : BaseGeometry | None
        See `fetch_gedi_l2a`.
    tile_dir : Path | None
        See `fetch_gedi_l2a`.
    params : dict
        JSON-serializable request parameters recorded in `tile_dir`.
    progress : Progress | None
        See `fetch_gedi_l2a`.

    Returns
    -------
    geopandas.GeoDataFrame
        Every tile's footprints, sorted by acquisition time.

    Raises
    ------
    ValueError
        If `polygon` doesn't intersect `bbox`.
    """
    tiles = split_bbox(bbox, max_size_m)
    if polygon is not None:
        n_bbox_tiles = len(tiles)
        shapely.prepare(polygon)
        tiles = [tile for tile in tiles if polygon.intersects(shapely.box(*tile))]
        log.info(f"Kept {len(tiles)}/{n_bbox_tiles} tile(s) of bbox={bbox} intersecting the AOI")
        if not tiles:
            msg = f"AOI polygon doesn't intersect bbox={bbox}"
            log.error(msg)
            raise ValueError(msg)
    if tile_dir is not None:
        _check_tile_params(tile_dir, params)
    log.info(
        f"Requesting GEDI {product} footprints over bbox={bbox}, "
        f"time_range={params['time_range']} in {len(tiles)} tile(s)"
    )

    gdfs = []
    with (
        temporary_task(progress, f"Fetching GEDI {product} tiles", total=len(tiles))
        if progress is not None
        else nullcontext(None)
    ) as task:
        n_reused = 0
        sliderule_ready = False
        for i, tile in enumerate(tiles, start=1):
            path = _tile_path(tile_dir, tile) if tile_dir is not None else None
            if path is not None and path.exists():
                tile_gdf = gpd.read_parquet(path)
                n_reused += 1
                log.debug(
                    f"GEDI {product} tile reused, {len(tile_gdf)} footprint(s) [{i}/{len(tiles)}]"
                )
            else:
                if not sliderule_ready:
                    sliderule.init(verbose=False)
                    sliderule_ready = True
                tile_gdf = fetch_tile(tile)
                if path is not None:
                    write_geoparquet(tile_gdf, path)
                log.info(
                    f"GEDI {product} tile fetched, {len(tile_gdf)} footprint(s) [{i}/{len(tiles)}]"
                )
            gdfs.append(tile_gdf)
            if progress is not None and task is not None:
                progress.advance(task)
    if n_reused:
        log.info(f"Reused {n_reused}/{len(tiles)} tile(s) already saved in {tile_dir}")

    # Concatenating an empty, possibly column-less tile frame would otherwise
    # degrade the other tiles' column dtypes to object.
    non_empty = [gdf for gdf in gdfs if not gdf.empty]
    if not non_empty:
        gdf = gdfs[0]
    else:
        gdf = gpd.GeoDataFrame(pd.concat(non_empty), crs=non_empty[0].crs).sort_index()
    log.info(f"Received {len(gdf)} footprint(s)")
    return gdf


def fetch_gedi_l2a(
    bbox: tuple[float, float, float, float],
    time_range: tuple[str, str] | None = None,
    fields: Sequence[str] | None = GEDI_L2A_DEFAULT_FIELDS,
    anc_fields: Sequence[str] | None = None,
    max_size_m: float = GEDI_MAX_TILE_SIZE_M,
    polygon: BaseGeometry | None = None,
    rh_percentiles: Sequence[int] | None = None,
    quality_filter: bool = True,
    tile_dir: Path | None = None,
    progress: Progress | None = None,
) -> gpd.GeoDataFrame:
    """
    Fetch GEDI L2A footprints over an AOI via SlideRule's on-demand subsetting.

    The AOI is split into tiles of at most `max_size_m` on each side (see
    `split_bbox`), fetched one request at a time. With `polygon` set, only the tiles
    intersecting it are requested - whole tiles are kept or skipped, so footprints
    outside `polygon` but inside a kept tile are still returned. With `tile_dir` set,
    each tile's result is saved there as soon as it's fetched and reused by a later
    call, so an interrupted run resumes where it stopped. A tile whose request failed
    server-side comes back empty (see `_select_columns`) and is saved like a genuinely
    empty one, so a resumed run doesn't retry it.

    With `quality_filter` set, footprints are filtered like geefetch's GEDI L2A
    collection: `quality_flag == 1`, `degrade_flag == 0`, `solar_elevation <= 0`
    (night shots), `sensitivity >= 0.9` and `0 <= rh98 <= 80`.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Bounding box (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
    time_range : tuple[str, str] | None
        (t0, t1) UTC timestamps (`'%Y-%m-%dT%H:%M:%SZ'`) bounding the granules
        processed. Defaults to None, which processes every GEDI L2A granule
        intersecting `bbox` across the whole mission archive (2019-present).
    fields : Sequence[str] | None
        Footprint columns to keep, a subset of SlideRule's fixed `gedi02ap` schema
        (`'orbit'`, `'solar_elevation'`, `'track'`, `'elevation_lm'`,
        `'elevation_hr'`, `'sensitivity'`, `'flags'`, `'beam'`). Defaults to
        `GEDI_L2A_DEFAULT_FIELDS` (ground elevation and canopy top only). Pass None
        to keep every column. `anc_fields` are always kept regardless of this
        filter.
    anc_fields : Sequence[str] | None
        Extra per-shot fields to read directly out of the source L2A granule,
        beyond `gedi02ap`'s fixed schema, via SlideRule's `anc_fields` request
        parameter - e.g. `'rh'` (the 101-element relative-height percentile
        array, see `rh_percentiles`), `'quality_flag'`, `'degrade_flag'`,
        `'surface_flag'`, `'digital_elevation_model'`, `'elevation_bias_flag'`,
        `'energy_total'`, `'num_detectedmodes'`, `'selected_algorithm'`,
        `'selected_mode'`, `'selected_mode_flag'`, `'delta_time'`,
        `'solar_azimuth'`, or a subgroup-qualified path like
        `'land_cover_data/landsat_treecover'`/`'land_cover_data/modis_treecover'`.
        Requesting `'shot_number'` this way is known broken - SlideRule's Python
        client mis-flattens it (`ValueError` on a length mismatch), confirmed
        live 2026-09-22. Defaults to None (none requested).
    max_size_m : float
        Maximum tile width and height, in meters. Defaults to
        `GEDI_MAX_TILE_SIZE_M`.
    polygon : BaseGeometry | None
        Exact AOI shape in EPSG:4326 (e.g. a union of country boundaries, whose
        bounding box is `bbox`), used to skip tiles that don't intersect it. Raises
        `ValueError` if it doesn't intersect `bbox` at all. Defaults to None (every
        tile of `bbox` is requested).
    rh_percentiles : Sequence[int] | None
        Relative-height percentiles to return as `rh{p}` columns, each in
        `[0, 100]`, sliced out of GEDI L2A's 101-element `rh` array (requested
        automatically). The raw array is not returned, even if `'rh'` is in
        `anc_fields`. Defaults to None (no `rh{p}` columns).
    quality_filter : bool
        Drop low-quality footprints, as geefetch does (see above). Defaults to True.
    tile_dir : Path | None
        Directory holding one GeoParquet file per completed tile, plus the request
        parameters they were fetched with. Raises `ValueError` if called with
        different parameters. Defaults to None (nothing saved, an interrupted run
        starts over).
    progress : Progress | None
        Rich progress tracker, advanced once per fetched tile. Defaults to None (no
        progress bar).

    Returns
    -------
    geopandas.GeoDataFrame
        One row per footprint, indexed by acquisition time, geometry in EPSG:7912
        (GEDI's standard CRS, ITRF2014).
    """
    params = {
        "bbox": bbox,
        "time_range": time_range,
        "fields": fields,
        "anc_fields": anc_fields,
        "max_size_m": max_size_m,
        "rh_percentiles": rh_percentiles,
        "quality_filter": quality_filter,
    }
    return _fetch_tiles(
        "L2A",
        lambda tile: _fetch_l2a_tile(
            tile, bbox, time_range, fields, anc_fields, rh_percentiles, quality_filter
        ),
        bbox,
        max_size_m,
        polygon,
        tile_dir,
        params,
        progress,
    )


def fetch_gedi_l4a(
    bbox: tuple[float, float, float, float],
    time_range: tuple[str, str] | None = None,
    fields: Sequence[str] | None = GEDI_L4A_DEFAULT_FIELDS,
    anc_fields: Sequence[str] | None = None,
    max_size_m: float = GEDI_MAX_TILE_SIZE_M,
    polygon: BaseGeometry | None = None,
    quality_filter: bool = True,
    tile_dir: Path | None = None,
    progress: Progress | None = None,
) -> gpd.GeoDataFrame:
    """
    Fetch GEDI L4A (aboveground biomass density) footprints over an AOI via
    SlideRule's on-demand subsetting.

    Tiled, polygon-restricted and resumable exactly like `fetch_gedi_l2a`.

    With `quality_filter` set, footprints are filtered like geefetch's GEDI L4A
    collection: `l4_quality_flag == 1`, `degrade_flag == 0` and `sensitivity >= 0.9`.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Bounding box (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
    time_range : tuple[str, str] | None
        (t0, t1) UTC timestamps (`'%Y-%m-%dT%H:%M:%SZ'`) bounding the granules
        processed. Defaults to None, which processes every GEDI L4A granule
        intersecting `bbox` across the whole mission archive (2019-present).
    fields : Sequence[str] | None
        Footprint columns to keep, a subset of SlideRule's fixed `gedi04ap` schema
        (`'agbd'`, `'elevation'`, `'sensitivity'`, `'solar_elevation'`, `'orbit'`,
        `'track'`, `'beam'`, `'flags'`). Defaults to `GEDI_L4A_DEFAULT_FIELDS`
        (biomass density and ground elevation only). Pass None to keep every column.
        `anc_fields` are always kept regardless of this filter.
    anc_fields : Sequence[str] | None
        Extra per-shot fields to read directly out of the source L4A granule via
        SlideRule's `anc_fields` request parameter - e.g. `'agbd_se'`,
        `'l4_quality_flag'`, `'degrade_flag'`. Defaults to None (none requested).
    max_size_m : float
        Maximum tile width and height, in meters. Defaults to
        `GEDI_MAX_TILE_SIZE_M`.
    polygon : BaseGeometry | None
        See `fetch_gedi_l2a`. Defaults to None (every tile of `bbox` is requested).
    quality_filter : bool
        Drop low-quality footprints, as geefetch does (see above). Defaults to True.
    tile_dir : Path | None
        See `fetch_gedi_l2a`. Defaults to None (nothing saved, an interrupted run
        starts over).
    progress : Progress | None
        Rich progress tracker, advanced once per fetched tile. Defaults to None (no
        progress bar).

    Returns
    -------
    geopandas.GeoDataFrame
        One row per footprint, indexed by acquisition time, geometry in EPSG:7912
        (GEDI's standard CRS, ITRF2014).
    """
    params = {
        "bbox": bbox,
        "time_range": time_range,
        "fields": fields,
        "anc_fields": anc_fields,
        "max_size_m": max_size_m,
        "quality_filter": quality_filter,
    }
    return _fetch_tiles(
        "L4A",
        lambda tile: _fetch_l4a_tile(tile, bbox, time_range, fields, anc_fields, quality_filter),
        bbox,
        max_size_m,
        polygon,
        tile_dir,
        params,
        progress,
    )


def _tile_path(tile_dir: Path, tile: tuple[float, float, float, float]) -> Path:
    """
    Path of the file holding one tile's saved footprints.

    Parameters
    ----------
    tile_dir : Path
        See `fetch_gedi_l2a`.
    tile : tuple[float, float, float, float]
        Tile (min_lon, min_lat, max_lon, max_lat), as returned by `split_bbox`.

    Returns
    -------
    Path
        `tile_dir/<hash>.parquet`, `<hash>` being a short digest of `tile`'s bounds
        rounded to 1e-6 degrees.
    """
    key = "_".join(f"{v:.6f}" for v in tile)
    return tile_dir / f"{hashlib.sha1(key.encode()).hexdigest()[:12]}.parquet"


def _check_tile_params(tile_dir: Path, params: dict) -> None:
    """
    Record the request parameters in `tile_dir`, or check they match those recorded.

    Parameters
    ----------
    tile_dir : Path
        See `fetch_gedi_l2a`.
    params : dict
        JSON-serializable request parameters.

    Raises
    ------
    ValueError
        If `tile_dir` already records different parameters.
    """
    path = tile_dir / "params.json"
    params = json.loads(json.dumps(params))
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != params:
            msg = (
                f"{tile_dir} holds tiles fetched with {saved}, not {params}; delete it to "
                "start over with the new parameters"
            )
            log.error(msg)
            raise ValueError(msg)
        return
    tile_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=tile_dir, prefix=".params.json.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(params, f)
    Path(tmp_name).replace(path)


def expand_rh(
    gdf: gpd.GeoDataFrame, percentiles: Sequence[int] = GEDI_L2A_RH_PERCENTILES
) -> gpd.GeoDataFrame:
    """
    Split GEDI L2A's `rh` ancillary field into flat `rh{p}` columns.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Result of `fetch_gedi_l2a` called with `'rh'` in `anc_fields` - each row's
        `rh` value is a 101-element array where index i is the i-th relative-height
        percentile.
    percentiles : Sequence[int]
        Percentiles to pull out as named columns, each in `[0, 100]`. Defaults to
        `GEDI_L2A_RH_PERCENTILES`.

    Returns
    -------
    gpd.GeoDataFrame
        `gdf` with the raw `rh` column replaced by one `rh{p}` column per
        `percentiles` entry.
    """
    for p in percentiles:
        gdf[f"rh{p}"] = gdf["rh"].apply(lambda arr, p=p: arr[p])
    return gdf.drop(columns="rh")
