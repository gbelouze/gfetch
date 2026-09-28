import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pystac
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import from_origin

from gfetch import ocm as ocm_module
from gfetch.ocm import ocm_grid, with_ocm_asset, write_mask, write_ocms

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


def _bands_item(tmp_path: Path, nir: np.ndarray) -> pystac.Item:
    full = np.full(nir.shape, 1000, dtype="uint16")
    return _item(
        {
            "red": _write_band(tmp_path / "red.tif", full),
            "green": _write_band(tmp_path / "green.tif", full),
            "nir": _write_band(tmp_path / "nir.tif", nir),
        }
    )


def test_write_ocms_writes_classes_with_nodata_at_20m(tmp_path: Path, monkeypatch) -> None:
    nir = np.full((8, 8), 1000, dtype="uint16")
    nir[:2, :2] = 0  # one 20 m pixel of no-data in `nir` only
    item = _bands_item(tmp_path, nir)
    seen: list[dict] = []

    def predict_from_array(array: np.ndarray, **kwargs: object) -> np.ndarray:
        seen.append({"shape": array.shape, **kwargs})
        return np.full((1, *array.shape[1:]), 2, dtype="uint8")

    fake = SimpleNamespace(predict_from_array=predict_from_array)
    monkeypatch.setattr(ocm_module, "_omnicloudmask", lambda: fake)
    path = tmp_path / "ocm.tif"

    done = list(
        write_ocms([(item, path)], models=["m"], device="cpu", dtype="float32", batch_size=3)
    )

    assert done == [item]
    assert seen[0]["shape"] == (3, 4, 4)
    assert seen[0]["custom_models"] == ["m"]
    assert seen[0]["batch_size"] == 3
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


def test_write_ocms_yields_every_item_in_order(tmp_path: Path, monkeypatch) -> None:
    items = []
    for i in range(5):
        (tmp_path / str(i)).mkdir()
        items.append(_bands_item(tmp_path / str(i), np.full((4, 4), 1000, dtype="uint16")))
        items[-1].id = str(i)
    monkeypatch.setattr(
        ocm_module, "predict", lambda bands, **k: np.zeros(bands.shape[1:], dtype="uint8")
    )
    jobs = [(item, tmp_path / item.id / "ocm.tif") for item in items]

    done = list(write_ocms(jobs, models=[], device="cpu", dtype="float32", batch_size=1))

    assert [item.id for item in done] == ["0", "1", "2", "3", "4"]
    assert all(path.exists() for _, path in jobs)


def test_write_mask_leaves_no_partial_file_on_failure(tmp_path: Path, monkeypatch) -> None:
    def failing_replace(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(ocm_module.os, "replace", failing_replace)

    with pytest.raises(OSError, match="disk full"):
        write_mask(
            tmp_path / "ocm.tif", np.zeros((2, 2), dtype="uint8"), _TRANSFORM, CRS.from_epsg(32736)
        )

    assert list(tmp_path.iterdir()) == []
