from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from gfetch.finalize import remove_cache
from gfetch.write import prepare_template, stack_bands, write_region


@pytest.fixture
def dataset() -> xr.Dataset:
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {
            "red": (("y", "x"), rng.random((20, 16)).astype("float32")),
            "green": (("y", "x"), rng.random((20, 16)).astype("float32")),
        },
        coords={"y": np.arange(20), "x": np.arange(16), "spatial_ref": 0},
    )


def _store(path: Path, ds: xr.Dataset, *, complete: bool = True) -> Path:
    ds = stack_bands(ds, ["red", "green"])
    prepare_template(ds.chunk({"x": 8, "y": 20}), path)
    write_region(ds.isel(x=slice(0, 8)), path, {"x": slice(0, 8), "y": slice(None)})
    if complete:
        write_region(ds.isel(x=slice(8, 16)), path, {"x": slice(8, 16), "y": slice(None)})
    return path


def _cache(path: Path) -> Path:
    for item in ("item-a", "item-b"):
        (path / item).mkdir(parents=True)
        (path / item / "red.tif").write_bytes(b"x")
        (path / item / "red.complete").touch()
    return path


def test_remove_cache_removes_everything_once_stores_complete(
    tmp_path: Path, dataset: xr.Dataset
) -> None:
    cache = _cache(tmp_path / "cache")
    cached_items = tmp_path / "cached_items.json"
    cached_items.write_text("{}")
    stores = [_store(tmp_path / "a.zarr", dataset), _store(tmp_path / "b.zarr", dataset)]

    remove_cache(cache, stores, cached_items_path=cached_items)

    assert not cache.exists()
    assert not cached_items.exists()


def test_remove_cache_refuses_when_a_store_is_incomplete(
    tmp_path: Path, dataset: xr.Dataset
) -> None:
    cache = _cache(tmp_path / "cache")
    cached_items = tmp_path / "cached_items.json"
    cached_items.write_text("{}")
    stores = [
        _store(tmp_path / "a.zarr", dataset),
        _store(tmp_path / "b.zarr", dataset, complete=False),
        tmp_path / "missing.zarr",
    ]

    with pytest.raises(ValueError, match=r"b\.zarr.*missing\.zarr"):
        remove_cache(cache, stores, cached_items_path=cached_items)

    assert (cache / "item-a" / "red.tif").exists()
    assert cached_items.exists()


def test_remove_cache_refuses_without_stores(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "cache")

    with pytest.raises(ValueError, match="No stores"):
        remove_cache(cache, [])

    assert cache.exists()
