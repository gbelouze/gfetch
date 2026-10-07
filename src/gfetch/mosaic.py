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
import shapely
import xarray as xr
from odc.geo.crs import CRS
from odc.geo.geobox import GeoBox
from odc.geo.geom import BoundingBox, Geometry, bbox_intersection
from pyproj.database import query_utm_crs_info
from shapely.geometry.base import BaseGeometry

from gfetch.utils.memory import log_chunk_footprint

log = logging.getLogger(__name__)

# A training read of a window fetches every chunk it touches, see `claude/tech-stack.md`'s
# "Training reads: stacked bands, larger chunks" section.
_DEFAULT_CHUNKS: dict[str, int] = {"x": 256, "y": 256}

# `composite`'s default `median` isn't chunk-wise associative, so dask must gather an
# entire spatial chunk's `time` axis into one chunk before it can reduce. If `time`
# arrives already split into several chunks (e.g. one per `groupby` group), dask
# inserts a rechunk step to consolidate them first, which copies every chunk through
# `dask.array.chunk.getitem` (split) and `np.concatenate` (merge). A single full-length
# `time` chunk from the start avoids that rechunk entirely.
_DEFAULT_TIME_CHUNK = -1

# 16x16 chunks per shard: 4096 px shards at the default chunk size. The mosaic stage
# writes one shard per task, so a shard's output (all bands, float32) is held in memory
# at once: 1 GB for 16 bands.
_DEFAULT_SHARD_FACTOR = 16

# 4x4 chunks per dask chunk: 1024 px at the default chunk size.
_DEFAULT_COMPUTE_CHUNK_FACTOR = 4

ORBIT_STATES: tuple[str, ...] = ("ascending", "descending")

# An AOI strip narrower than this in an end UTM zone goes to its neighbour zone
# instead of getting its own store. At the equator, 1 degree past a zone's edge is
# 4 degrees from its central meridian: scale error ~0.20%, vs ~0.10% at the edge.
_MIN_ZONE_WIDTH_DEG = 1.0

# An item's EPSG:4326 `geometry` only approximates its footprint (straight edges in
# lon/lat are curved in the native CRS, and providers may simplify it), so an item
# grazing a geobox's edge could otherwise test as disjoint from it. The same holds for
# an EPSG:4326 AOI polygon reprojected onto a geobox.
_FOOTPRINT_PAD_PX = 2


__all__ = [
    "ORBIT_STATES",
    "FootprintIndex",
    "composite",
    "group_by_utm_zone",
    "load",
    "mask_clouds",
    "mask_nodata",
    "mosaic",
    "orbit_state_variables",
    "outside_aoi",
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
    chunks: dict[str, int], compute_chunk_factor: int | None, shards: dict[str, int] | None
) -> dict[str, int]:
    """
    Resolve the dask chunk sizes the `mosaic` stage computes with.

    Parameters
    ----------
    chunks : dict[str, int]
        Store chunk sizes, as resolved by `resolve_chunks`.
    compute_chunk_factor : int | None
        Number of store chunks per dask chunk along each spatial dimension. None uses
        `_DEFAULT_COMPUTE_CHUNK_FACTOR`.
    shards : dict[str, int] | None
        Shard sizes, as resolved by `resolve_shards`, or None if unsharded.

    Returns
    -------
    dict[str, int]
        `chunks` with its spatial sizes multiplied by the compute chunk factor.

    Raises
    ------
    ValueError
        If a dask chunk doesn't evenly divide the unit of work (a shard, or a chunk
        when unsharded), which would let two dask chunks share a written file.
    """
    factor = (
        compute_chunk_factor if compute_chunk_factor is not None else _DEFAULT_COMPUTE_CHUNK_FACTOR
    )
    compute = {**chunks, **{dim: factor * chunks[dim] for dim in ("y", "x")}}
    unit = shards if shards is not None else chunks
    for dim in ("y", "x"):
        if unit[dim] % compute[dim]:
            raise ValueError(
                f"compute_chunk_factor={factor} gives {compute[dim]} px dask "
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
    patch_url: Callable[[str], str] | None = None,
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
    patch_url : Callable[[str], str] | None
        Applied to every asset href before loading, e.g. to sign it, see
        `gfetch.sources.planetary_computer_signer`. Hrefs are patched here, while
        pixels are only read once the dataset is computed. Defaults to None (hrefs
        used as-is).

    Returns
    -------
    xr.Dataset
        Lazy, dask-backed dataset with one data variable per band.
    """
    # Without this, GDAL/rasterio falls through to botocore's full credential chain on
    # every S3 asset, hanging on an EC2-instance-metadata lookup that never succeeds
    # off-EC2. Every gfetch STAC source is either an unsigned public S3 bucket or Azure
    # Blob Storage, signed via `patch_url`, so `aws_unsigned` is always safe here.
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
        patch_url=patch_url,
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


class FootprintIndex:
    """
    Spatial index answering which items a geobox's pixels can draw from.

    An item's EPSG:4326 `geometry` only approximates its footprint, so a geobox is
    padded by `_FOOTPRINT_PAD_PX` pixels before testing, see that constant. Items
    without a `geometry` are assumed to cover every geobox.

    Parameters
    ----------
    items : Sequence[pystac.Item]
        Candidate items.
    """

    def __init__(self, items: Sequence[pystac.Item]) -> None:
        self._items = list(items)
        self._unlocated: list[int] = []
        self._located: list[int] = []
        footprints = []
        for i, item in enumerate(self._items):
            if item.geometry is None:
                self._unlocated.append(i)
            else:
                self._located.append(i)
                footprints.append(shapely.geometry.shape(item.geometry))
        self._tree = shapely.STRtree(footprints)

    def intersecting(self, geobox: GeoBox) -> list[pystac.Item]:
        """
        List the items whose footprint intersects `geobox`.

        Parameters
        ----------
        geobox : GeoBox
            Target pixel grid.

        Returns
        -------
        list[pystac.Item]
            Items whose `geometry` intersects `geobox` padded by `_FOOTPRINT_PAD_PX`
            pixels, plus every item without a `geometry`, in their original order.
            odc-stac fuses same-group items in the order it's given them.
        """
        extent = geobox.pad(_FOOTPRINT_PAD_PX).extent.to_crs("EPSG:4326", wrapdateline=True)
        hits = self._tree.query(extent.geom, predicate="intersects")
        kept = sorted([*self._unlocated, *(self._located[i] for i in hits)])
        return [self._items[i] for i in kept]


def _items_intersecting(items: Sequence[pystac.Item], geobox: GeoBox) -> list[pystac.Item]:
    """
    Keep only the items whose footprint intersects `geobox`, see `FootprintIndex`.

    odc-stac builds the time axis from every item it's given, so a shard loaded with its
    whole zone's items carries every time step of the zone. Steps no item covers are
    never read, but are still filled with nodata, held in memory, masked and reduced.

    Parameters
    ----------
    items : Sequence[pystac.Item]
        Candidate items.
    geobox : GeoBox
        Target pixel grid.

    Returns
    -------
    list[pystac.Item]
        `FootprintIndex(items).intersecting(geobox)`. If that's empty, the first item
        alone, which loads a single all-nodata time step.
    """
    kept = FootprintIndex(items).intersecting(geobox)
    log.debug(f"{len(kept)}/{len(items)} item(s) intersect the geobox")
    return kept if kept else list(items[:1])


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
    patch_url: Callable[[str], str] | None = None,
) -> xr.Dataset:
    """
    Load, cloud-mask, and composite STAC items into a single mosaic.

    Convenience wrapper chaining `load`, `mask_nodata`, `mask_clouds`, and
    `composite`. A pixel with no valid observation is NaN. Items whose footprint
    doesn't intersect `geobox` are dropped before loading, see `_items_intersecting`.

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
    patch_url : Callable[[str], str] | None
        Passed to `load`. Defaults to None.

    Returns
    -------
    xr.Dataset
        Lazy, dask-backed single-timestep mosaic.
    """
    items = _items_intersecting(items, geobox)
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
            patch_url=patch_url,
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


def _utm_zone_number(crs: CRS) -> int:
    """
    UTM zone number of a UTM CRS, regardless of its hemisphere.

    Parameters
    ----------
    crs : CRS
        A UTM CRS, as returned by `odc.geo.crs.CRS.utm`.

    Returns
    -------
    int
        Zone number, from 1 to 60.
    """
    utm_zone = crs.proj.utm_zone
    assert utm_zone is not None, f"{crs} is not a UTM CRS"
    return int(utm_zone[:-1])


def _utm_zone_lon_band(zone_number: int) -> tuple[float, float]:
    """
    Longitude band (EPSG:4326) covered by a UTM zone.

    Parameters
    ----------
    zone_number : int
        UTM zone number, from 1 to 60.

    Returns
    -------
    tuple[float, float]
        `(west, east)` longitude bounds of the zone's natural 6-degree-wide band.
    """
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


def _zone_extents(aoi: BoundingBox) -> dict[CRS, BoundingBox]:
    """
    Split an AOI between the UTM zones it spans, each zone getting one output grid.

    Each zone gets the AOI clipped to its natural longitude band, except that an end
    zone's strip narrower than `_MIN_ZONE_WIDTH_DEG` goes to its neighbour zone, whose
    extent then reaches past its own band. The narrower end is merged first, until
    both ends are wide enough or a single zone is left. Zones are merged by number,
    so both hemispheres of an AOI crossing the equator are merged alike.

    Parameters
    ----------
    aoi : BoundingBox
        AOI bounding box in EPSG:4326.

    Returns
    -------
    dict[CRS, BoundingBox]
        Each remaining zone's share of `aoi`, in EPSG:4326.
    """
    zones = _utm_zones_for_aoi(aoi)
    columns: list[list[float]] = []
    numbers: list[int] = []
    for number in sorted({_utm_zone_number(crs) for crs in zones}):
        west, east = _utm_zone_lon_band(number)
        columns.append([max(west, aoi.left), min(east, aoi.right)])
        numbers.append(number)
    while len(columns) > 1:
        widths = [columns[0][1] - columns[0][0], columns[-1][1] - columns[-1][0]]
        end = 0 if widths[0] <= widths[1] else -1
        if widths[end] >= _MIN_ZONE_WIDTH_DEG:
            break
        west, east = columns.pop(end)
        number = numbers.pop(end)
        # The popped zone's neighbour is now the end on the same side.
        columns[end] = [min(columns[end][0], west), max(columns[end][1], east)]
        log.debug(
            f"Merging the AOI's {east - west:.2f}-degree strip in UTM zone {number} into "
            f"zone {numbers[end]}"
        )
    extents = dict(zip(numbers, columns, strict=True))
    return {
        crs: BoundingBox(extents[n][0], aoi.bottom, extents[n][1], aoi.top, crs="EPSG:4326")
        for crs in zones
        if (n := _utm_zone_number(crs)) in extents
    }


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

    A zone's extent is its share of the AOI, see `_zone_extents`: a thin strip of the
    AOI in an end zone is assigned to its neighbour zone instead.

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
        appears in more than one list; a zone the AOI spans but no item overlaps, or
        whose strip was merged into its neighbour, is omitted.
    """
    groups: dict[CRS, list[pystac.Item]] = {}
    for crs, zone_extent in _zone_extents(BoundingBox(*aoi_bbox, crs="EPSG:4326")).items():
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
    Build the output pixel grid for one UTM zone: its share of the AOI, see
    `_zone_extents`.

    Parameters
    ----------
    crs : CRS
        Target UTM CRS for this zone, one of `group_by_utm_zone`'s keys for the same
        `aoi_bbox`.
    aoi_bbox : tuple[float, float, float, float]
        Full AOI bounding box (min_lon, min_lat, max_lon, max_lat) in EPSG:4326;
        may span more than one UTM zone.
    resolution : float
        Output pixel resolution, in `crs`'s units (meters, for UTM).

    Returns
    -------
    GeoBox
        Pixel grid covering `crs`'s share of `aoi_bbox`.
    """
    aoi = BoundingBox(*aoi_bbox, crs="EPSG:4326")
    # GeoBox.from_bbox() only reprojects when crs is literally the string "utm";
    # a resolved CRS object is instead taken as the CRS the bbox values are already
    # in, so the intersection (computed in EPSG:4326) must be reprojected explicitly
    # first.
    extent = _zone_extents(aoi)[crs].to_crs(crs)
    return GeoBox.from_bbox(extent, resolution=resolution)


def outside_aoi(geobox: GeoBox, aoi: BaseGeometry) -> Callable[[dict[str, slice]], bool]:
    """
    Build a test for whether a region of `geobox` lies entirely outside an AOI.

    Parameters
    ----------
    geobox : GeoBox
        Pixel grid the regions index into.
    aoi : BaseGeometry
        Exact AOI shape, in EPSG:4326.

    Returns
    -------
    Callable[[dict[str, slice]], bool]
        Takes a region as `{"y": slice, "x": slice}` and returns True if that region,
        padded by `_FOOTPRINT_PAD_PX` pixels, doesn't intersect `aoi`.
    """
    extent = geobox.extent.to_crs("EPSG:4326", wrapdateline=True)
    clipped = Geometry(aoi, "EPSG:4326").intersection(extent)
    if clipped.is_empty:
        return lambda region: True
    local = clipped.to_crs(geobox.crs).geom
    shapely.prepare(local)

    def outside(region: dict[str, slice]) -> bool:
        sub = cast("GeoBox", geobox[region["y"], region["x"]])
        return not local.intersects(sub.pad(_FOOTPRINT_PAD_PX).extent.geom)

    return outside
