import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from gfetch.vrt import vrt_path, write_vrt
from gfetch.write import prepare_template, write, write_region

_GEOBOX = GeoBox.from_bbox((500000, 4000000, 500160, 4000080), crs="EPSG:32631", resolution=10)


def _dataset() -> xr.Dataset:
    ds = xr.Dataset(
        {
            name: (("y", "x"), np.full(_GEOBOX.shape, value, dtype="float32"))
            for name, value in [("red", 1.0), ("green", 2.0)]
        },
        coords=xr_coords(_GEOBOX, crs_coord_name="spatial_ref"),
    )
    ds["green"][:2, :2] = np.nan
    return ds


def test_write_vrt_stacks_variables_as_bands(tmp_path: Path) -> None:
    store = tmp_path / "mosaic_epsg32631.zarr"
    write(_dataset().chunk({"y": 4, "x": 4}), store)

    dest = write_vrt(store, ["green", "red"])

    assert dest == vrt_path(store) == tmp_path / "mosaic_epsg32631.vrt"
    with rasterio.open(dest) as src:
        assert src.count == 2
        assert src.descriptions == ("green", "red")
        assert src.crs.to_epsg() == 32631
        assert src.transform == _GEOBOX.affine
        assert src.nodatavals == (pytest.approx(np.nan, nan_ok=True),) * 2
        green, red = src.read()
    np.testing.assert_array_equal(green, _dataset()["green"].values)
    assert (red == 1.0).all()


def test_write_vrt_survives_moving_with_its_store(tmp_path: Path) -> None:
    store = tmp_path / "a" / "mosaic.zarr"
    write(_dataset().chunk({"y": 4, "x": 4}), store)
    write_vrt(store, ["red"])

    moved = tmp_path / "b"
    shutil.move(store.parent, moved)

    with rasterio.open(moved / "mosaic.vrt") as src:
        assert (src.read(1) == 1.0).all()


def test_write_vrt_reads_unwritten_region_as_nodata(tmp_path: Path) -> None:
    store = tmp_path / "mosaic.zarr"
    ds = _dataset()
    prepare_template(ds.chunk({"y": 8, "x": 8}), store)
    write_region(ds.isel(x=slice(0, 8)), store, {"x": slice(0, 8), "y": slice(None)})

    write_vrt(store, ["red"])

    with rasterio.open(vrt_path(store)) as src:
        red = src.read(1)
    assert (red[:, :8] == 1.0).all()
    assert np.isnan(red[:, 8:]).all()


def test_write_vrt_declares_sharded_store_inner_chunks_as_blocks(tmp_path: Path) -> None:
    """rasterio's bundled GDAL can't read sharded Zarr, so only the XML is checked."""
    store = tmp_path / "mosaic.zarr"
    prepare_template(
        _dataset().chunk({"y": 8, "x": 8}), store, shards={"y": 8, "x": 8}, chunks={"y": 4, "x": 4}
    )

    write_vrt(store, ["red"])

    props = ET.parse(vrt_path(store)).getroot().find("VRTRasterBand/SimpleSource/SourceProperties")
    assert props is not None
    assert (props.get("BlockXSize"), props.get("BlockYSize")) == ("4", "4")


def test_write_vrt_refuses_store_without_crs(tmp_path: Path) -> None:
    store = tmp_path / "mosaic.zarr"
    ds = xr.Dataset(
        {"red": (("y", "x"), np.ones((4, 4), dtype="float32"))},
        coords={"y": np.arange(4), "x": np.arange(4)},
    )
    write(ds, store)

    with pytest.raises(ValueError, match="no CRS"):
        write_vrt(store, ["red"])

    assert not vrt_path(store).exists()
