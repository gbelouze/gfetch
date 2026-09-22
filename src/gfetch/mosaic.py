"""Load/mosaic stage: load STAC items onto a common grid, cloud-mask, and composite.

Runs on compute nodes, no internet required, as long as items' asset hrefs already
point at a local cache (see `gfetch.download`) - or directly against remote hrefs for
single-machine, internet-connected use.
"""

import logging
from collections.abc import Sequence
from typing import Literal, cast

import odc.stac
import pystac
import xarray as xr
from odc.geo.crs import CRS
from odc.geo.geobox import GeoBox
from odc.geo.geom import BoundingBox, bbox_intersection
from pyproj.database import query_utm_crs_info

from gfetch.utils.memory import log_chunk_footprint

log = logging.getLogger(__name__)

_DEFAULT_CHUNKS: dict[str, int] = {"x": 2048, "y": 2048}

# `composite`'s default `median` isn't chunk-wise associative, so dask must gather an
# entire spatial chunk's `time` axis into one chunk before it can reduce - if `time`
# arrives already split into several chunks (e.g. one per `groupby` group), dask
# inserts a rechunk step to consolidate them first, which copies every chunk through
# `dask.array.chunk.getitem` (split) and `np.concatenate` (merge). A single full-length
# `time` chunk from the start avoids that rechunk entirely.
_DEFAULT_TIME_CHUNK = -1


__all__ = [
    "composite",
    "group_by_utm_zone",
    "load",
    "mask_clouds",
    "mosaic",
    "mosaic_by_zone",
    "zone_geobox",
]


def load(
    items: Sequence[pystac.Item],
    geobox: GeoBox,
    bands: Sequence[str],
    *,
    groupby: str = "solar_day",
    chunks: dict[str, int] | None = None,
    resampling: str | dict[str, str] | None = None,
) -> xr.Dataset:
    """
    Load STAC items onto a common grid, lazily (dask-backed).

    Parameters
    ----------
    items : Sequence[pystac.Item]
        Items to load, as returned by `gfetch.search.search` or `gfetch.download.
        download_items`.
    geobox : GeoBox
        Target pixel grid (CRS, resolution, extent) to reproject/resample onto.
    bands : Sequence[str]
        Asset keys to load.
    groupby : str
        odc-stac grouping strategy for merging same-group overlapping scenes before
        compositing. Defaults to 'solar_day', recommended for wide-AOI Sentinel-2
        mosaics over stacking every individual scene as a separate time step.
    chunks : dict[str, int] | None
        Dask chunk sizes, e.g. `{"time": 1, "x": 512, "y": 512}`. Defaults to None,
        which chunks the spatial dims at `{"x": 2048, "y": 2048}` - passing `None`
        through to odc-stac itself would instead load everything eagerly, without
        Dask. Regardless of this argument, `time` itself defaults to a single
        full-length chunk unless explicitly given here.
    resampling : str | dict[str, str] | None
        Resampling method, passed straight through to `odc.stac.load`. Either one
        method for every band, or a per-band `dict[str, str]` (a `"*"` key sets the
        default for bands not otherwise listed). Defaults to None, which falls back to
        odc-stac's own default ('nearest' for every band).

    Returns
    -------
    xr.Dataset
        Lazy, dask-backed dataset with one data variable per band.
    """
    # Without this, GDAL/rasterio falls through to botocore's full credential chain on
    # every S3 asset, hanging on an EC2-instance-metadata lookup that never succeeds
    # off-EC2. Every gfetch STAC source is either an unsigned public S3 bucket or Azure
    # Blob Storage with its SAS token already embedded in the href, so `aws_unsigned`
    # is always safe here.
    odc.stac.configure_s3_access(aws_unsigned=True)
    log.debug(f"Loading {len(items)} item(s), bands={list(bands)}, geobox shape={geobox.shape}")
    effective_chunks = dict(chunks) if chunks is not None else dict(_DEFAULT_CHUNKS)
    effective_chunks.setdefault("time", _DEFAULT_TIME_CHUNK)
    ds = odc.stac.load(
        items,
        bands=list(bands),
        geobox=geobox,
        groupby=groupby,
        chunks=cast("dict[str, int | Literal['auto']]", effective_chunks),
        resampling=resampling,
    )
    log.info(f"Loaded dataset: {dict(ds.sizes)}")
    log_chunk_footprint(ds, log)
    return ds


def mask_clouds(ds: xr.Dataset, mask_band: str, mask_out: frozenset[int]) -> xr.Dataset:
    """
    Mask out invalid/cloudy pixels using a per-pixel classification band.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset as returned by `load`, including the classification band named
        `mask_band`.
    mask_band : str
        Data variable holding the per-pixel classification (e.g. Sentinel-2's SCL).
    mask_out : frozenset[int]
        Classification values to mask out (set to NaN) as invalid/cloudy.

    Returns
    -------
    xr.Dataset
        `ds` without `mask_band`, with masked-out pixels set to NaN in every
        remaining data variable.
    """
    valid = ~ds[mask_band].isin(list(mask_out))
    data_vars = [v for v in ds.data_vars if v != mask_band]
    return ds[data_vars].where(valid)


def composite(ds: xr.Dataset, *, dim: str = "time", method: str = "median") -> xr.Dataset:
    """
    Reduce a time-stacked dataset to a single composite.

    Parameters
    ----------
    ds : xr.Dataset
        Time-stacked dataset, e.g. as returned by `mask_clouds`.
    dim : str
        Dimension to reduce over. Defaults to 'time'.
    method : str
        Name of the `xr.Dataset` reduction method to use (e.g. 'median', 'mean').
        Defaults to 'median'.

    Returns
    -------
    xr.Dataset
        `ds` reduced over `dim`, skipping NaNs.
    """
    reducer = getattr(ds, method)
    return reducer(dim=dim, skipna=True)


def _pin_mask_band_resampling(
    resampling: str | dict[str, str] | None, mask_band: str | None
) -> str | dict[str, str] | None:
    """
    Force `mask_band` to 'nearest' resampling, regardless of `resampling`.

    A classification band (e.g. Sentinel-2's SCL) holds categorical values -
    interpolating them with anything but nearest-neighbor produces class values that
    were never in the source data.

    Parameters
    ----------
    resampling : str | dict[str, str] | None
        Resampling method requested for the data bands, as passed to `mosaic`.
    mask_band : str | None
        Classification band to pin to 'nearest', or None if there is none.

    Returns
    -------
    str | dict[str, str] | None
        `resampling` unchanged if `mask_band` is None, otherwise a `dict[str, str]`
        with `mask_band` forced to 'nearest'.
    """
    if mask_band is None:
        return resampling
    if resampling is None:
        per_band: dict[str, str] = {}
    elif isinstance(resampling, dict):
        per_band = dict(resampling)
    else:
        per_band = {"*": resampling}
    per_band[mask_band] = "nearest"
    return per_band


def mosaic(
    items: Sequence[pystac.Item],
    geobox: GeoBox,
    bands: Sequence[str],
    *,
    mask_band: str | None = None,
    mask_out: frozenset[int] = frozenset(),
    groupby: str = "solar_day",
    chunks: dict[str, int] | None = None,
    method: str = "median",
    resampling: str | dict[str, str] | None = None,
) -> xr.Dataset:
    """
    Load, cloud-mask, and composite STAC items into a single mosaic.

    Convenience wrapper chaining `load`, `mask_clouds`, and `composite`.

    Parameters
    ----------
    items : Sequence[pystac.Item]
        Items to mosaic.
    geobox : GeoBox
        Target pixel grid to reproject/resample onto.
    bands : Sequence[str]
        Asset keys to load and composite. If `mask_band` is given and not already in
        `bands`, it is loaded too and dropped after masking.
    mask_band : str | None
        Classification band used for cloud masking. Defaults to None (no masking).
    mask_out : frozenset[int]
        Classification values to mask out. Unused if `mask_band` is None.
    groupby : str
        odc-stac grouping strategy, passed to `load`. Defaults to 'solar_day'.
    chunks : dict[str, int] | None
        Dask chunk sizes, passed to `load`. Defaults to None.
    method : str
        Composite reduction method, passed to `composite`. Defaults to 'median'.
    resampling : str | dict[str, str] | None
        Resampling method for the data bands, passed to `load`. `mask_band`, if given,
        is always loaded with 'nearest' resampling regardless of this setting, since it
        holds categorical values. Defaults to None (odc-stac's own default).

    Returns
    -------
    xr.Dataset
        Lazy, dask-backed single-timestep mosaic.
    """
    load_bands = list(bands)
    if mask_band is not None and mask_band not in load_bands:
        load_bands.append(mask_band)

    ds = load(
        items,
        geobox,
        load_bands,
        groupby=groupby,
        chunks=chunks,
        resampling=_pin_mask_band_resampling(resampling, mask_band),
    )
    if mask_band is not None:
        ds = mask_clouds(ds, mask_band, mask_out)
    return composite(ds, method=method)


def _utm_zone_lon_band(crs: CRS) -> tuple[float, float]:
    """
    Longitude band (EPSG:4326) covered by a UTM zone.

    Parameters
    ----------
    crs : CRS
        A UTM CRS, as returned by `odc.geo.crs.CRS.utm`.

    Returns
    -------
    tuple[float, float]
        `(west, east)` longitude bounds of the zone's natural 6-degree-wide band.
    """
    utm_zone = crs.proj.utm_zone
    assert utm_zone is not None, f"{crs} is not a UTM CRS"
    zone_number = int(utm_zone[:-1])
    west = -180.0 + 6.0 * (zone_number - 1)
    return west, west + 6.0


def _utm_zones_for_aoi(aoi: BoundingBox) -> list[CRS]:
    """
    Enumerate every UTM zone whose area of use overlaps an AOI.

    Unlike `odc.geo.crs.CRS.utm`, which picks the single best-fit zone for a bbox, this
    returns every candidate - an AOI spanning several zones needs all of them, not just
    the best match.

    Parameters
    ----------
    aoi : BoundingBox
        AOI bounding box in EPSG:4326.

    Returns
    -------
    list[CRS]
        Candidate UTM zones (correct hemisphere per latitude), in pyproj's match order.
    """
    return [
        CRS(f"{info.auth_name}:{info.code}")
        for info in query_utm_crs_info(datum_name="WGS 84", area_of_interest=aoi.aoi)
    ]


def _zone_aoi_extent(crs: CRS, aoi: BoundingBox) -> BoundingBox:
    """
    AOI clipped to a UTM zone's natural longitude band, still in EPSG:4326.

    Parameters
    ----------
    crs : CRS
        UTM zone to clip the AOI to.
    aoi : BoundingBox
        AOI bounding box in EPSG:4326.

    Returns
    -------
    BoundingBox
        `aoi` intersected with `crs`'s 6-degree-wide longitude band, in EPSG:4326.
    """
    west, east = _utm_zone_lon_band(crs)
    zone_band = BoundingBox(west, aoi.bottom, east, aoi.top, crs="EPSG:4326")
    return bbox_intersection([aoi, zone_band])


def _bbox_overlaps(a: BoundingBox, b: BoundingBox) -> bool:
    """
    Check whether two bounding boxes overlap with non-zero area.

    Parameters
    ----------
    a : BoundingBox
        First bounding box, same CRS as `b`.
    b : BoundingBox
        Second bounding box, same CRS as `a`.

    Returns
    -------
    bool
        True if `a` and `b` overlap with non-zero area (merely touching doesn't count).
    """
    inter = bbox_intersection([a, b])
    return inter.left < inter.right and inter.bottom < inter.top


def group_by_utm_zone(
    items: Sequence[pystac.Item], aoi_bbox: tuple[float, float, float, float]
) -> dict[CRS, list[pystac.Item]]:
    """
    Assign items to every UTM zone (among those the AOI spans) their footprint overlaps.

    An item is not assumed to fit within a single zone - that holds for Sentinel-2's
    MGRS-tiled items, but not in general (e.g. Sentinel-1 GRD items are delivered in
    EPSG:4326 and routinely span several zones). Grouping by AOI-relevant zone overlap
    instead of one "native" zone per item keeps a wide item's contribution from being
    silently dropped from a zone its footprint actually covers, while still producing
    the same result as a one-zone-per-item assignment whenever every item does fit in
    a single zone.

    Parameters
    ----------
    items : Sequence[pystac.Item]
        Items to group, e.g. as returned by `gfetch.search.search`.
    aoi_bbox : tuple[float, float, float, float]
        Full AOI bounding box (min_lon, min_lat, max_lon, max_lat) in EPSG:4326 -
        determines which UTM zones are even candidates.

    Returns
    -------
    dict[CRS, list[pystac.Item]]
        Items grouped by overlapping UTM zone. An item spanning more than one zone
        appears in more than one list; a zone the AOI spans but no item overlaps is
        omitted.
    """
    aoi = BoundingBox(*aoi_bbox, crs="EPSG:4326")
    groups: dict[CRS, list[pystac.Item]] = {}
    for crs in _utm_zones_for_aoi(aoi):
        zone_extent = _zone_aoi_extent(crs, aoi)
        zone_items = [
            item
            for item in items
            if item.bbox is not None
            and _bbox_overlaps(zone_extent, BoundingBox(*item.bbox, crs="EPSG:4326"))
        ]
        if zone_items:
            groups[crs] = zone_items
    return groups


def zone_geobox(crs: CRS, aoi_bbox: tuple[float, float, float, float], resolution: float) -> GeoBox:
    """
    Build the output pixel grid for one UTM zone: the AOI clipped to that zone's own
    natural longitude band.

    Parameters
    ----------
    crs : CRS
        Target UTM CRS for this zone.
    aoi_bbox : tuple[float, float, float, float]
        Full AOI bounding box (min_lon, min_lat, max_lon, max_lat) in EPSG:4326 -
        may span more than one UTM zone.
    resolution : float
        Output pixel resolution, in `crs`'s units (meters, for UTM).

    Returns
    -------
    GeoBox
        Pixel grid covering the portion of `aoi_bbox` that falls within `crs`'s zone.
    """
    aoi = BoundingBox(*aoi_bbox, crs="EPSG:4326")
    # GeoBox.from_bbox() only reprojects when crs is literally the string "utm" - a
    # resolved CRS object is instead taken as the CRS the bbox values are already in,
    # so the intersection (computed in EPSG:4326) must be reprojected explicitly first.
    extent = _zone_aoi_extent(crs, aoi).to_crs(crs)
    return GeoBox.from_bbox(extent, resolution=resolution)


def mosaic_by_zone(
    items: Sequence[pystac.Item],
    aoi_bbox: tuple[float, float, float, float],
    bands: Sequence[str],
    *,
    resolution: float,
    mask_band: str | None = None,
    mask_out: frozenset[int] = frozenset(),
    groupby: str = "solar_day",
    chunks: dict[str, int] | None = None,
    method: str = "median",
    resampling: str | dict[str, str] | None = None,
) -> dict[CRS, xr.Dataset]:
    """
    Mosaic items into one composite per UTM zone the AOI spans.

    Items are grouped by which zone(s) their own footprint overlaps (see
    `group_by_utm_zone`) rather than reprojected into one AOI-wide zone chosen from the
    AOI's centroid - a country-scale AOI spanning several zones produces one dataset per
    zone instead of warping everything into a single arbitrarily-chosen one. An item
    whose own footprint spans more than one zone (not possible for Sentinel-2's
    MGRS-tiled items, routine for e.g. Sentinel-1) contributes to every zone it
    overlaps.

    Parameters
    ----------
    items : Sequence[pystac.Item]
        Items to mosaic, e.g. as returned by `gfetch.search.search`.
    aoi_bbox : tuple[float, float, float, float]
        Full AOI bounding box (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
    bands : Sequence[str]
        Asset keys to load and composite, passed to `mosaic`.
    resolution : float
        Output pixel resolution, in each zone's own UTM CRS units (meters).
    mask_band : str | None
        Classification band used for cloud masking, passed to `mosaic`. Defaults to
        None (no masking).
    mask_out : frozenset[int]
        Classification values to mask out, passed to `mosaic`. Defaults to an empty
        frozenset.
    groupby : str
        odc-stac grouping strategy, passed to `mosaic`. Defaults to 'solar_day'.
    chunks : dict[str, int] | None
        Dask chunk sizes, passed to `mosaic`. Defaults to None.
    method : str
        Composite reduction method, passed to `mosaic`. Defaults to 'median'.
    resampling : str | dict[str, str] | None
        Resampling method for the data bands, passed to `mosaic`. `mask_band`, if
        given, is always loaded with 'nearest' resampling regardless of this setting.
        Defaults to None (odc-stac's own default).

    Returns
    -------
    dict[CRS, xr.Dataset]
        One lazy, dask-backed mosaic per UTM zone spanned by `items`.
    """
    zones = group_by_utm_zone(items, aoi_bbox)
    log.info(f"Items span {len(zones)} UTM zone(s): {[crs.epsg for crs in zones]}")

    result: dict[CRS, xr.Dataset] = {}
    for crs, zone_items in zones.items():
        log.debug(f"{crs}: {len(zone_items)} item(s)")
        result[crs] = mosaic(
            zone_items,
            zone_geobox(crs, aoi_bbox, resolution),
            bands,
            mask_band=mask_band,
            mask_out=mask_out,
            groupby=groupby,
            chunks=chunks,
            method=method,
            resampling=resampling,
        )
    return result
