"""GDAL VRT descriptors presenting a Zarr mosaic store as a single multi-band raster.

GDAL's Zarr driver exposes each band array as its own subdataset, so QGIS shows a
store one band at a time. A VRT stacks them into one dataset, without copying any
pixel data.
"""

import logging
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import odc.geo.xr  # noqa: F401  registers the `.odc` accessor
import xarray as xr
import zarr

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


def write_vrt(store: Path, variables: Sequence[str]) -> Path:
    """
    Write a VRT next to `store` stacking its `variables` as the bands of one raster.

    Each band's source is the array directory itself (e.g. `mosaic.zarr/B04`),
    relative to the VRT, so the VRT stays valid when moved or copied together with
    its store. Georeferencing comes from the store's `x`/`y` coordinates and is
    written into the VRT, so it doesn't depend on GDAL resolving the store's grid
    mapping. Reading the VRT needs a GDAL that can read the store, i.e. GDAL >= 3.13
    for a sharded one. The store may still be incomplete: unwritten shards read as
    nodata. Overwrites any previous VRT, atomically.

    Parameters
    ----------
    store : Path
        Zarr store directory, already initialized via
        `gfetch.write.prepare_template`.
    variables : Sequence[str]
        Data variables to expose, in band order. Each must be a 2D `(y, x)` array.

    Returns
    -------
    Path
        Path of the VRT, see `vrt_path`.

    Raises
    ------
    ValueError
        If the store has no recoverable CRS, or a variable's dtype has no GDAL
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
    group = zarr.open_group(store, mode="r")
    for i, name in enumerate(variables, start=1):
        za = group[name]
        assert isinstance(za, zarr.Array), f"{name} is not an array"
        assert za.shape == (height, width), f"{name} isn't on the store's (y, x) grid"
        dtype = _GDAL_DTYPES.get(str(za.dtype))
        if dtype is None:
            msg = f"{store}: {name}'s dtype {za.dtype} has no GDAL equivalent"
            log.error(msg)
            raise ValueError(msg)
        band = ET.SubElement(root, "VRTRasterBand", dataType=dtype, band=str(i))
        ET.SubElement(band, "Description").text = name
        nodata = _nodata(za.metadata.fill_value)
        if nodata is not None:
            ET.SubElement(band, "NoDataValue").text = nodata
        source = ET.SubElement(band, "SimpleSource")
        ET.SubElement(source, "SourceFilename", relativeToVRT="1").text = f"{store.name}/{name}"
        ET.SubElement(source, "SourceBand").text = "1"
        block_y, block_x = za.chunks
        # Lets GDAL defer opening each band's array until its pixels are read.
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
    log.info(f"Wrote {dest} ({len(variables)} band(s) from {store.name})")
    return dest
