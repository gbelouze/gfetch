"""GEDI vector fetch via SlideRule: on-demand, server-side subsetting of remote GEDI
HDF5 granules (https://slideruleearth.io). Unlike gfetch's raster pipeline, this needs
no separate search/download stage: SlideRule resolves the matching granules via CMR and
performs the byte-range subsetting itself, so one request returns an already-subsetted
GeoDataFrame of footprints. No credentials are required against SlideRule's public
cluster (the default) - NASA Earthdata authentication against the source DAAC is handled
server-side; it's only needed for a self-hosted SlideRule deployment.
"""

import logging
from collections.abc import Sequence

import geopandas as gpd
from sliderule import gedi, sliderule
from sliderule.session import Session

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

__all__ = ["GEDI_L2A_DEFAULT_FIELDS", "GEDI_L2A_RH_PERCENTILES", "expand_rh", "fetch_gedi_l2a"]

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


def fetch_gedi_l2a(
    bbox: tuple[float, float, float, float],
    time_range: tuple[str, str] | None = None,
    fields: Sequence[str] | None = GEDI_L2A_DEFAULT_FIELDS,
    anc_fields: Sequence[str] | None = None,
) -> gpd.GeoDataFrame:
    """
    Fetch GEDI L2A footprints over an AOI via SlideRule's on-demand subsetting.

    No filtering is applied beyond the AOI/time range: SlideRule's `degrade_filter`,
    `l2_quality_filter`, and `surface_filter` all default to off, so degraded/low-
    quality footprints are included as-is.

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
        array, see `expand_rh`), `'quality_flag'`, `'degrade_flag'`,
        `'surface_flag'`, `'digital_elevation_model'`, `'elevation_bias_flag'`,
        `'energy_total'`, `'num_detectedmodes'`, `'selected_algorithm'`,
        `'selected_mode'`, `'selected_mode_flag'`, `'delta_time'`,
        `'solar_azimuth'`, or a subgroup-qualified path like
        `'land_cover_data/landsat_treecover'`/`'land_cover_data/modis_treecover'`.
        Requesting `'shot_number'` this way is known broken - SlideRule's Python
        client mis-flattens it (`ValueError` on a length mismatch), confirmed
        live 2026-09-22. Defaults to None (none requested).

    Returns
    -------
    geopandas.GeoDataFrame
        One row per footprint, indexed by acquisition time, geometry in EPSG:7912
        (GEDI's standard CRS, ITRF2014).
    """
    sliderule.init(verbose=False)
    parms: dict = {"poly": _bbox_to_poly(bbox)}
    if time_range is not None:
        parms["t0"], parms["t1"] = time_range
    if anc_fields:
        parms["anc_fields"] = list(anc_fields)

    log.info(f"Requesting GEDI L2A footprints over bbox={bbox}, time_range={time_range}")
    gdf = gedi.gedi02ap(parms)
    log.info(f"Received {len(gdf)} footprint(s)")

    if fields is not None:
        keep = list(dict.fromkeys([*fields, *(anc_fields or ())]))
        missing = [f for f in keep if f not in gdf.columns]
        if missing:
            if not gdf.empty:
                raise KeyError(f"field(s) {missing} not in response columns {list(gdf.columns)}")
            # `sliderule.gedi.gedi02ap()` returns a bare-geometry frame with no
            # data columns at all (`sliderule.emptyframe()`) both when the AOI/time
            # range genuinely matches zero footprints and when the request itself
            # failed server-side (SlideRule logs that failure loudly on its own,
            # e.g. "Unexpected termination of response") - either way, this
            # column selection would otherwise raise a confusing KeyError instead
            # of a well-formed empty result.
            log.warning(
                f"Empty response with no data columns (missing {missing}) - the request may "
                "have failed server-side rather than genuinely matching zero footprints"
            )
            for f in missing:
                gdf[f] = []
        gdf = gdf[[*keep, gdf.geometry.name]]
    return gdf


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
