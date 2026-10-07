"""GDAL VRT descriptors presenting a Zarr mosaic store as a named multi-band raster.

GDAL's Zarr driver already opens a store's `(band, y, x)` array as a multi-band raster,
but leaves its bands unnamed (their names only appear as `DIM_band_VALUE` metadata). A
VRT names them, without copying any pixel data.
"""

import logging
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import cast

import numpy as np
import odc.geo.xr  # noqa: F401  registers the `.odc` accessor
import xarray as xr
import zarr

from gfetch.write import BAND_NAMES_ATTR, STACKED_VARIABLE

log = logging.getLogger(__name__)

__all__ = ["vrt_path", "write_vrt"]

_GDAL_DTYPES = {
    "uint8": "Byte",
    "int8": "Int8",
    "uint16": "UInt16",
    "int16": "Int16",
    "uint32": "UInt32",
    "int32": "Int32",
    "uint64": "UInt64",
    "int64": "Int64",
    "float32": "Float32",
    "float64": "Float64",
}


def vrt_path(store: Path) -> Path:
    """
    Path of the VRT `write_vrt` writes for `store`.

    Parameters
    ----------
    store : Path
        Zarr store directory, e.g. `mosaic_epsg32631.zarr`.

    Returns
    -------
    Path
        Sibling VRT path, e.g. `mosaic_epsg32631.vrt`.
    """
    return store.with_suffix(".vrt")


def _nodata(fill_value: object) -> str | None:
    """
    Format a Zarr fill value as a VRT `NoDataValue`.

    Parameters
    ----------
    fill_value : object
        The array's fill value.

    Returns
    -------
    str | None
        GDAL's spelling of the value, or None if the array has none.
    """
    if fill_value is None:
        return None
    value = fill_value.item() if isinstance(fill_value, np.generic) else fill_value
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return str(value)


def write_vrt(store: Path) -> Path:
    """
    Write a VRT next to `store` exposing its stacked bands as one named raster.

    Each band's source is its index in the stacked array's directory (e.g.
    `mosaic.zarr/bands`), relative to the VRT, so the VRT stays valid when moved or
    copied together with its store. Georeferencing comes from the store's `x`/`y` coordinates and is
    written into the VRT, so it doesn't depend on GDAL resolving the store's grid
    mapping. Reading the VRT needs a GDAL that can read the store, i.e. GDAL >= 3.13
    for a sharded one. The store may still be incomplete: unwritten shards read as
    nodata. Overwrites any previous VRT, atomically.

    Parameters
    ----------
    store : Path
        Zarr store directory holding a `gfetch.write.STACKED_VARIABLE` array, already
        initialized via `gfetch.write.prepare_template`.

    Returns
    -------
    Path
        Path of the VRT, see `vrt_path`.

    Raises
    ------
    ValueError
        If the store has no recoverable CRS, or its array's dtype has no GDAL
        equivalent.
    """
    geobox = xr.open_zarr(store, consolidated=False).odc.geobox
    if geobox is None or geobox.crs is None:
        msg = f"{store}: no CRS recoverable from its coordinates"
        log.error(msg)
        raise ValueError(msg)
    height, width = geobox.shape
    a = geobox.affine

    root = ET.Element("VRTDataset", rasterXSize=str(width), rasterYSize=str(height))
    ET.SubElement(root, "SRS", dataAxisToSRSAxisMapping="1,2").text = geobox.crs.to_wkt()
    ET.SubElement(root, "GeoTransform").text = ", ".join(
        repr(v) for v in (a.c, a.a, a.b, a.f, a.d, a.e)
    )
    za = zarr.open_group(store, mode="r")[STACKED_VARIABLE]
    assert isinstance(za, zarr.Array), f"{STACKED_VARIABLE} is not an array"
    names = [str(name) for name in cast("list[str]", za.attrs[BAND_NAMES_ATTR])]
    assert za.shape == (len(names), height, width), (
        f"{STACKED_VARIABLE} isn't a (band, y, x) array on the store's grid"
    )
    dtype = _GDAL_DTYPES.get(str(za.dtype))
    if dtype is None:
        msg = f"{store}: {STACKED_VARIABLE}'s dtype {za.dtype} has no GDAL equivalent"
        log.error(msg)
        raise ValueError(msg)
    nodata = _nodata(za.metadata.fill_value)
    _, block_y, block_x = za.chunks
    for i, name in enumerate(names, start=1):
        band = ET.SubElement(root, "VRTRasterBand", dataType=dtype, band=str(i))
        ET.SubElement(band, "Description").text = name
        if nodata is not None:
            ET.SubElement(band, "NoDataValue").text = nodata
        source = ET.SubElement(band, "SimpleSource")
        ET.SubElement(
            source, "SourceFilename", relativeToVRT="1"
        ).text = f"{store.name}/{STACKED_VARIABLE}"
        ET.SubElement(source, "SourceBand").text = str(i)
        # Lets GDAL defer opening the array until its pixels are read.
        ET.SubElement(
            source,
            "SourceProperties",
            RasterXSize=str(width),
            RasterYSize=str(height),
            DataType=dtype,
            BlockXSize=str(block_x),
            BlockYSize=str(block_y),
        )
    ET.indent(root)

    dest = vrt_path(store)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            ET.ElementTree(root).write(f, encoding="utf-8")
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    log.info(f"Wrote {dest} ({len(names)} band(s) from {store.name})")
    return dest
