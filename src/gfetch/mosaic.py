"""Load/mosaic stage: load STAC items onto a common grid, cloud-mask, and composite.

Runs on compute nodes, no internet required, as long as items' asset hrefs already
point at a local cache (see `gfetch.download`) - or directly against remote hrefs for
single-machine, internet-connected use.
"""

import logging
from collections.abc import Sequence

import odc.stac
import pystac
import xarray as xr
from odc.geo.crs import CRS
from odc.geo.geobox import GeoBox
from odc.geo.geom import BoundingBox, bbox_intersection

log = logging.getLogger(__name__)

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
    chunks: dict | None = None,
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
    chunks : dict | None
        Dask chunk sizes, e.g. `{"time": 1, "x": 512, "y": 512}`. Defaults to None,
        which uses odc-stac's own default chunking.

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
    ds = odc.stac.load(
        items, bands=list(bands), geobox=geobox, groupby=groupby, chunks=chunks or {}
    )
    log.info(f"Loaded dataset: {dict(ds.sizes)}")
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


def mosaic(
    items: Sequence[pystac.Item],
    geobox: GeoBox,
    bands: Sequence[str],
    *,
    mask_band: str | None = None,
    mask_out: frozenset[int] = frozenset(),
    groupby: str = "solar_day",
    chunks: dict | None = None,
    method: str = "median",
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
    chunks : dict | None
        Dask chunk sizes, passed to `load`. Defaults to None.
    method : str
        Composite reduction method, passed to `composite`. Defaults to 'median'.

    Returns
    -------
    xr.Dataset
        Lazy, dask-backed single-timestep mosaic.
    """
    load_bands = list(bands)
    if mask_band is not None and mask_band not in load_bands:
        load_bands.append(mask_band)

    ds = load(items, geobox, load_bands, groupby=groupby, chunks=chunks)
    if mask_band is not None:
        ds = mask_clouds(ds, mask_band, mask_out)
    return composite(ds, method=method)


def group_by_utm_zone(items: Sequence[pystac.Item]) -> dict[CRS, list[pystac.Item]]:
    """
    Group items by the UTM zone their own footprint naturally falls in.

    Each item's zone is resolved from its own STAC `bbox`, not the AOI as a whole -
    a single Sentinel-2/Landsat tile always fits within one UTM zone, so this reflects
    each item's actual native CRS.

    Parameters
    ----------
    items : Sequence[pystac.Item]
        Items to group, e.g. as returned by `gfetch.search.search`.

    Returns
    -------
    dict[CRS, list[pystac.Item]]
        Items grouped by native UTM CRS, in first-seen order.
    """
    groups: dict[CRS, list[pystac.Item]] = {}
    for item in items:
        assert item.bbox is not None
        crs = CRS.utm(BoundingBox(*item.bbox, crs="EPSG:4326"))
        groups.setdefault(crs, []).append(item)
    return groups


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
    west, east = _utm_zone_lon_band(crs)
    aoi = BoundingBox(*aoi_bbox, crs="EPSG:4326")
    zone_band = BoundingBox(west, aoi.bottom, east, aoi.top, crs="EPSG:4326")
    # GeoBox.from_bbox() only reprojects when crs is literally the string "utm" - a
    # resolved CRS object is instead taken as the CRS the bbox values are already in,
    # so the intersection (computed in EPSG:4326) must be reprojected explicitly first.
    extent = bbox_intersection([aoi, zone_band]).to_crs(crs)
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
    chunks: dict | None = None,
    method: str = "median",
) -> dict[CRS, xr.Dataset]:
    """
    Mosaic items into one composite per native UTM zone the AOI spans.

    Items are grouped by their own footprint's UTM zone rather than reprojected into
    one AOI-wide zone chosen from the AOI's centroid - a country-scale AOI spanning
    several zones produces one dataset per zone instead of warping everything into a
    single arbitrarily-chosen one.

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
    chunks : dict | None
        Dask chunk sizes, passed to `mosaic`. Defaults to None.
    method : str
        Composite reduction method, passed to `mosaic`. Defaults to 'median'.

    Returns
    -------
    dict[CRS, xr.Dataset]
        One lazy, dask-backed mosaic per UTM zone spanned by `items`.
    """
    zones = group_by_utm_zone(items)
    log.info(f"Items span {len(zones)} UTM zone(s): {[crs.epsg for crs in zones]}")
    return {
        crs: mosaic(
            zone_items,
            zone_geobox(crs, aoi_bbox, resolution),
            bands,
            mask_band=mask_band,
            mask_out=mask_out,
            groupby=groupby,
            chunks=chunks,
            method=method,
        )
        for crs, zone_items in zones.items()
    }
