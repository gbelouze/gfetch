"""Load/mosaic stage: load STAC items onto a common grid, cloud-mask, and composite.

Runs on compute nodes, no internet required, as long as items' asset hrefs already
point at a local cache (see `gfetch.download`), or directly against remote hrefs for
single-machine, internet-connected use.
"""

import logging
from collections.abc import Callable, Sequence
from typing import Literal, cast

import numpy as np
import odc.stac
import pystac
import xarray as xr
from odc.geo.crs import CRS
from odc.geo.geobox import GeoBox
from odc.geo.geom import BoundingBox, bbox_intersection
from pyproj.database import query_utm_crs_info

from gfetch.utils.memory import log_chunk_footprint

log = logging.getLogger(__name__)

_DEFAULT_CHUNKS: dict[str, int] = {"x": 256, "y": 256}

# `composite`'s default `median` isn't chunk-wise associative, so dask must gather an
# entire spatial chunk's `time` axis into one chunk before it can reduce. If `time`
# arrives already split into several chunks (e.g. one per `groupby` group), dask
# inserts a rechunk step to consolidate them first, which copies every chunk through
# `dask.array.chunk.getitem` (split) and `np.concatenate` (merge). A single full-length
# `time` chunk from the start avoids that rechunk entirely.
_DEFAULT_TIME_CHUNK = -1

# 32x32 chunks per shard: 8192 px shards at the default chunk size. The mosaic stage
# writes one shard per task, so a shard's output (all bands) is held in memory at once.
_DEFAULT_SHARD_FACTOR = 32

ORBIT_STATES: tuple[str, ...] = ("ascending", "descending")


__all__ = [
    "ORBIT_STATES",
    "composite",
    "group_by_utm_zone",
    "load",
    "mask_clouds",
    "mask_nodata",
    "mosaic",
    "orbit_state_variables",
    "resolve_chunks",
    "resolve_compute_chunks",
    "resolve_shards",
    "zone_geobox",
]


def resolve_chunks(chunks: dict[str, int] | None) -> dict[str, int]:
    """
    Resolve a user-supplied chunk dict against gfetch's own defaults.

    Pulled out of `load` so callers that need the actual chunk sizes without loading
    anything (e.g. to plan a store's shards) don't have to duplicate or guess at
    gfetch's defaults.

    Parameters
    ----------
    chunks : dict[str, int] | None
        Dask chunk sizes as passed to `load`/`mosaic`. Defaults to None, which uses
        `{"x": 256, "y": 256}`.

    Returns
    -------
    dict[str, int]
        `chunks` if given, else gfetch's own spatial defaults, always including a
        `"time"` entry (a single full-length chunk unless explicitly overridden); see
        `load`'s docstring for why.
    """
    effective = dict(chunks) if chunks is not None else dict(_DEFAULT_CHUNKS)
    effective.setdefault("time", _DEFAULT_TIME_CHUNK)
    return effective


def resolve_shards(shard_factor: int | None, chunks: dict[str, int]) -> dict[str, int] | None:
    """
    Resolve a shard factor into shard sizes, against gfetch's own defaults.

    Parameters
    ----------
    shard_factor : int | None
        Number of chunks per shard along each spatial dimension. Defaults to None,
        which uses `_DEFAULT_SHARD_FACTOR`.
    chunks : dict[str, int]
        Chunk sizes, as resolved by `resolve_chunks`.

    Returns
    -------
    dict[str, int] | None
        Shard size per spatial dimension, or None if `shard_factor` is 1 (no
        sharding).
    """
    factor = shard_factor if shard_factor is not None else _DEFAULT_SHARD_FACTOR
    if factor == 1:
        return None
    return {dim: factor * chunks[dim] for dim in ("y", "x")}


def resolve_compute_chunks(
    chunks: dict[str, int], compute_chunk_factor: int, shards: dict[str, int] | None
) -> dict[str, int]:
    """
    Resolve the dask chunk sizes the `mosaic` stage computes with.

    Parameters
    ----------
    chunks : dict[str, int]
        Store chunk sizes, as resolved by `resolve_chunks`.
    compute_chunk_factor : int
        Number of store chunks per dask chunk along each spatial dimension.
    shards : dict[str, int] | None
        Shard sizes, as resolved by `resolve_shards`, or None if unsharded.

    Returns
    -------
    dict[str, int]
        `chunks` with its spatial sizes multiplied by `compute_chunk_factor`.

    Raises
    ------
    ValueError
        If a dask chunk doesn't evenly divide the unit of work (a shard, or a chunk
        when unsharded), which would let two dask chunks share a written file.
    """
    compute = {**chunks, **{dim: compute_chunk_factor * chunks[dim] for dim in ("y", "x")}}
    unit = shards if shards is not None else chunks
    for dim in ("y", "x"):
        if unit[dim] % compute[dim]:
            raise ValueError(
                f"compute_chunk_factor={compute_chunk_factor} gives {compute[dim]} px dask "
                f"chunks along {dim!r}, which don't evenly divide the {unit[dim]} px "
                f"{'shard' if shards is not None else 'chunk'}; use a divisor of "
                "shard_factor"
            )
    return compute


def load(
    items: Sequence[pystac.Item],
    geobox: GeoBox,
    bands: Sequence[str],
    *,
    groupby: str = "solar_day",
    chunks: dict[str, int] | None = None,
    resampling: str | dict[str, str] | None = None,
    log_footprint: bool = True,
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
        which chunks the spatial dims at `{"x": 256, "y": 256}`; passing `None`
        through to odc-stac itself would instead load everything eagerly, without
        Dask. Regardless of this argument, `time` itself defaults to a single
        full-length chunk unless explicitly given here.
    resampling : str | dict[str, str] | None
        Resampling method, passed straight through to `odc.stac.load`. Either one
        method for every band, or a per-band `dict[str, str]` (a `"*"` key sets the
        default for bands not otherwise listed). Defaults to None, which falls back to
        odc-stac's own default ('nearest' for every band).
    log_footprint : bool
        Log the loaded dataset's per-chunk memory footprint, see
        `gfetch.utils.memory.log_chunk_footprint`. Defaults to True.

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
    effective_chunks = resolve_chunks(chunks)
    ds = odc.stac.load(
        items,
        bands=list(bands),
        geobox=geobox,
        groupby=groupby,
        chunks=cast("dict[str, int | Literal['auto']]", effective_chunks),
        resampling=resampling,
    )
    log.info(f"Loaded dataset: {dict(ds.sizes)}")
    if log_footprint:
        log_chunk_footprint(ds, log)
    return ds


def mask_nodata(ds: xr.Dataset, *, exclude: str | None = None) -> xr.Dataset:
    """
    Set each data variable's `nodata` pixels (from its `nodata` attribute) to NaN.

    Without this, a reduction like `composite`'s median counts nodata as a real value:
    a pixel covered by only some time steps is pulled towards it, and one covered by
    none comes out as `nodata` instead of NaN.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset as returned by `load`, whose variables carry odc-stac's `nodata`
        attribute.
    exclude : str | None
        Data variable left untouched (e.g. a classification band that
        `mask_clouds` still needs as integers). Defaults to None.

    Returns
    -------
    xr.Dataset
        `ds` with `nodata` pixels set to NaN and the `nodata` attribute dropped, in
        every variable that had one other than `exclude`.
    """

    def mask(da: xr.DataArray) -> xr.DataArray:
        if da.name == exclude or "nodata" not in da.attrs:
            return da
        masked = da.where(da != da.attrs["nodata"])
        masked.attrs = {k: v for k, v in da.attrs.items() if k != "nodata"}
        return masked

    return xr.Dataset({name: mask(da) for name, da in ds.data_vars.items()}, attrs=ds.attrs)


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

    A classification band (e.g. Sentinel-2's SCL) holds categorical values;
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
    log_footprint: bool = True,
    on_load: Callable[[xr.Dataset], object] | None = None,
    split_orbit_states: bool = False,
) -> xr.Dataset:
    """
    Load, cloud-mask, and composite STAC items into a single mosaic.

    Convenience wrapper chaining `load`, `mask_nodata`, `mask_clouds`, and
    `composite`. A pixel with no valid observation is NaN.

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
        Dask chunk sizes, passed to `load`. A `"time"` entry other than a single
        full-length chunk is overridden with a logged warning (see `resolve_chunks`'s
        `_DEFAULT_TIME_CHUNK` note) - `mosaic` always reduces over the whole time
        axis via `method`, so a smaller one is never beneficial and forces dask to
        rechunk internally before reducing, which can also shrink the spatial `x`/`y`
        chunk size as a side effect (confirmed live 2026-09-22, see
        `claude/tasks.md`). Defaults to None.
    method : str
        Composite reduction method, passed to `composite`. Defaults to 'median'.
    resampling : str | dict[str, str] | None
        Resampling method for the data bands, passed to `load`. `mask_band`, if given,
        is always loaded with 'nearest' resampling regardless of this setting, since it
        holds categorical values. Defaults to None (odc-stac's own default).
    log_footprint : bool
        Passed to `load`. Defaults to True.
    on_load : Callable[[xr.Dataset], object] | None
        Called with the lazily loaded dataset, before masking and compositing (e.g.
        to measure how many bytes the mosaic processes). Called once per orbit
        state with `split_orbit_states`. Defaults to None.
    split_orbit_states : bool
        Composite each `sat:orbit_state` (see `ORBIT_STATES`) separately, into
        variables named as `orbit_state_variables` does. An orbit state with no
        items gets all-NaN variables; an item with no `sat:orbit_state` among
        `ORBIT_STATES` raises `ValueError`. Defaults to False (one composite over
        every item).

    Returns
    -------
    xr.Dataset
        Lazy, dask-backed single-timestep mosaic.
    """
    load_bands = list(bands)
    if mask_band is not None and mask_band not in load_bands:
        load_bands.append(mask_band)

    resolved_chunks = resolve_chunks(chunks)
    if resolved_chunks["time"] != _DEFAULT_TIME_CHUNK:
        log.warning(
            f"chunks['time']={resolved_chunks['time']!r} requested, but mosaic() always "
            f"reduces over the whole time axis via {method}() - a non-full time chunk is "
            "never beneficial here and forces dask to rechunk internally before reducing, "
            "which can also shrink the spatial x/y chunk size as a side effect, breaking "
            "alignment with any pre-planned Zarr write region. Overriding to a single "
            f"full-length chunk ({_DEFAULT_TIME_CHUNK})."
        )
        resolved_chunks["time"] = _DEFAULT_TIME_CHUNK

    def build(group: Sequence[pystac.Item]) -> xr.Dataset:
        ds = load(
            group,
            geobox,
            load_bands,
            groupby=groupby,
            chunks=resolved_chunks,
            resampling=_pin_mask_band_resampling(resampling, mask_band),
            log_footprint=log_footprint,
        )
        if on_load is not None:
            on_load(ds)
        ds = mask_nodata(ds, exclude=mask_band)
        if mask_band is not None:
            ds = mask_clouds(ds, mask_band, mask_out)
        return composite(ds, method=method)

    result = _composite_by_orbit_state(items, build) if split_orbit_states else build(items)

    # Belt-and-suspenders: the `time` override above removes the known trigger for
    # composite()'s reduction disturbing the x/y chunk grid, but nothing guarantees
    # that grid is preserved exactly for every variable regardless (e.g. a band
    # needing extra resampling to reach the common geobox). write_region()'s Zarr
    # requires exact alignment for every variable, so we pin it explicitly.
    return result.chunk({"y": resolved_chunks["y"], "x": resolved_chunks["x"]})


def orbit_state_variables(bands: Sequence[str]) -> list[str]:
    """
    Name the variables `mosaic(split_orbit_states=True)` produces.

    Parameters
    ----------
    bands : Sequence[str]
        Data bands, as passed to `mosaic`.

    Returns
    -------
    list[str]
        `{band}_{orbit_state}` for every band and orbit state in `ORBIT_STATES`,
        grouped by band (e.g. `['vv_ascending', 'vv_descending', ...]`).
    """
    return [f"{band}_{state}" for band in bands for state in ORBIT_STATES]


def _composite_by_orbit_state(
    items: Sequence[pystac.Item], build: Callable[[Sequence[pystac.Item]], xr.Dataset]
) -> xr.Dataset:
    """
    Composite each orbit state's items separately and merge them into one dataset.

    Parameters
    ----------
    items : Sequence[pystac.Item]
        Items to mosaic, each with a `sat:orbit_state` in `ORBIT_STATES`.
    build : Callable[[Sequence[pystac.Item]], xr.Dataset]
        Loads and composites a non-empty list of items.

    Returns
    -------
    xr.Dataset
        One `{band}_{orbit_state}` variable per band and orbit state, all-NaN for an
        orbit state with no items.

    Raises
    ------
    ValueError
        If an item has no `sat:orbit_state` among `ORBIT_STATES`.
    """
    groups: dict[str, list[pystac.Item]] = {state: [] for state in ORBIT_STATES}
    for item in items:
        state = item.properties.get("sat:orbit_state")
        if state not in groups:
            msg = f"Item {item.id} has sat:orbit_state={state!r}, expected one of {ORBIT_STATES}"
            log.error(msg)
            raise ValueError(msg)
        groups[state].append(item)
    log.debug(f"Items per orbit state: { {s: len(g) for s, g in groups.items()} }")

    composites = {state: build(group) for state, group in groups.items() if group}
    template = next(iter(composites.values()))
    parts = []
    for state in ORBIT_STATES:
        if state in composites:
            ds = composites[state]
        else:
            log.info(f"No {state} items, its variables are all NaN")
            ds = xr.full_like(template, np.nan)
        parts.append(ds.rename({v: f"{v}_{state}" for v in ds.data_vars}))

    # all parts share the same geobbox so "override" is necessary
    merged = xr.merge(parts, compat="override", join="exact")
    return merged[orbit_state_variables([str(v) for v in template.data_vars])]


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
    returns every candidate: an AOI spanning several zones needs all of them.

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

    An item is not assumed to fit within a single zone; that holds for Sentinel-2's
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
        Full AOI bounding box (min_lon, min_lat, max_lon, max_lat) in EPSG:4326,
        which determines which UTM zones are even candidates.

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
        Full AOI bounding box (min_lon, min_lat, max_lon, max_lat) in EPSG:4326;
        may span more than one UTM zone.
    resolution : float
        Output pixel resolution, in `crs`'s units (meters, for UTM).

    Returns
    -------
    GeoBox
        Pixel grid covering the portion of `aoi_bbox` that falls within `crs`'s zone.
    """
    aoi = BoundingBox(*aoi_bbox, crs="EPSG:4326")
    # GeoBox.from_bbox() only reprojects when crs is literally the string "utm";
    # a resolved CRS object is instead taken as the CRS the bbox values are already
    # in, so the intersection (computed in EPSG:4326) must be reprojected explicitly
    # first.
    extent = _zone_aoi_extent(crs, aoi).to_crs(crs)
    return GeoBox.from_bbox(extent, resolution=resolution)
