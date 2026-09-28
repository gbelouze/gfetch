import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pystac
import pytest
import rasterio
from affine import Affine
from rasterio.transform import from_origin

from gfetch import ocm as ocm_module
from gfetch.ocm import ocm_grid, with_ocm_asset, write_ocm

_TRANSFORM = from_origin(600000, 9500020, 10, 10)


def _item(assets: dict[str, pystac.Asset] | None = None) -> pystac.Item:
    item = pystac.Item(
        id="item",
        geometry=None,
        bbox=None,
        datetime=datetime.datetime(2024, 7, 19, tzinfo=datetime.UTC),
        properties={},
    )
    for key, asset in (assets or {}).items():
        item.add_asset(key, asset)
    return item


def _write_band(path: Path, data: np.ndarray) -> pystac.Asset:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="uint16",
        nodata=0,
        crs="EPSG:32736",
        transform=_TRANSFORM,
    ) as dst:
        dst.write(data, 1)
    return pystac.Asset(href=str(path))


def test_ocm_grid_covers_the_same_extent_at_20m() -> None:
    shape, transform = ocm_grid((10980, 10980), _TRANSFORM)

    assert shape == (5490, 5490)
    assert transform == from_origin(600000, 9500020, 20, 20)


def test_with_ocm_asset_derives_its_grid_from_red() -> None:
    red = pystac.Asset(
        href="red.tif",
        extra_fields={"proj:shape": [10980, 10980], "proj:transform": list(_TRANSFORM)[:6]},
    )
    item = _item({"red": red})

    result = with_ocm_asset(item, Path("/cache/item/ocm.tif"))

    asset = result.assets["ocm"]
    assert asset.href == "/cache/item/ocm.tif"
    assert asset.extra_fields["proj:shape"] == [5490, 5490]
    assert Affine(*asset.extra_fields["proj:transform"]) == from_origin(600000, 9500020, 20, 20)
    assert asset.extra_fields["raster:bands"][0]["nodata"] == 255
    assert "ocm" not in item.assets


def test_write_ocm_writes_classes_with_nodata_at_20m(tmp_path: Path, monkeypatch) -> None:
    data = np.full((8, 8), 1000, dtype="uint16")
    data[:2, :2] = 0  # one 20 m pixel of no-data in `nir` only
    red = _write_band(tmp_path / "red.tif", np.full((8, 8), 1000, dtype="uint16"))
    green = _write_band(tmp_path / "green.tif", np.full((8, 8), 1000, dtype="uint16"))
    nir = _write_band(tmp_path / "nir.tif", data)
    item = _item({"red": red, "green": green, "nir": nir})
    seen: list[np.ndarray] = []

    def predict_from_array(array: np.ndarray, **kwargs: object) -> np.ndarray:
        seen.append(array)
        return np.full((1, *array.shape[1:]), 2, dtype="uint8")

    fake = SimpleNamespace(predict_from_array=predict_from_array)
    monkeypatch.setattr(ocm_module, "_omnicloudmask", lambda: fake)
    path = tmp_path / "ocm.tif"

    write_ocm(item, path, model_dir=tmp_path / "models")

    assert seen[0].shape == (3, 4, 4)
    with rasterio.open(path) as src:
        assert src.nodata == 255
        assert src.res == (20, 20)
        classes = src.read(1)
    expected = np.full((4, 4), 2, dtype="uint8")
    expected[0, 0] = 255
    np.testing.assert_array_equal(classes, expected)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "green.tif",
        "nir.tif",
        "ocm.tif",
        "red.tif",
    ]


def test_write_ocm_leaves_no_partial_file_on_failure(tmp_path: Path, monkeypatch) -> None:
    bands = {
        key: _write_band(tmp_path / f"{key}.tif", np.full((4, 4), 1000, dtype="uint16"))
        for key in ("red", "green", "nir")
    }
    monkeypatch.setattr(ocm_module, "predict", lambda *a, **k: np.zeros((2, 2), dtype="uint8"))

    def failing_replace(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(ocm_module.os, "replace", failing_replace)
    path = tmp_path / "ocm.tif"

    with pytest.raises(OSError, match="disk full"):
        write_ocm(_item(bands), path, model_dir=tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["green.tif", "nir.tif", "red.tif"]
