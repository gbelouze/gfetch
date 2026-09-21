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
from odc.geo.geobox import GeoBox

log = logging.getLogger(__name__)

__all__ = ["composite", "load", "mask_clouds", "mosaic"]


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
