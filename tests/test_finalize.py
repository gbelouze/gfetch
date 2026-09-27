from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from zarr.storage import ZipStore

from gfetch.finalize import pack_store, packed_store_path, remove_cache
from gfetch.write import prepare_template, write_region


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


def test_pack_store_round_trips(tmp_path: Path, dataset: xr.Dataset) -> None:
    store = _store(tmp_path / "mosaic.zarr", dataset)

    dest = pack_store(store, ["red", "green"])

    assert dest == packed_store_path(store) == tmp_path / "mosaic.zarr.zip"
    assert store.exists()
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []
    packed = xr.open_zarr(ZipStore(dest, mode="r"), consolidated=False)
    xr.testing.assert_equal(packed[["red", "green"]].load(), dataset[["red", "green"]])


def test_pack_store_refuses_incomplete_store(tmp_path: Path, dataset: xr.Dataset) -> None:
    store = _store(tmp_path / "mosaic.zarr", dataset, complete=False)

    with pytest.raises(ValueError, match="incomplete"):
        pack_store(store, ["red", "green"])

    assert not packed_store_path(store).exists()


def test_pack_store_counts_all_nodata_region_as_written(tmp_path: Path) -> None:
    """An all-fill-value region must still be written as chunks, or the store could
    never count as complete.
    """
    ds = xr.Dataset(
        {"red": (("y", "x"), np.full((20, 16), np.nan, dtype="float32"))},
        coords={"y": np.arange(20), "x": np.arange(16)},
    )
    store = _store(tmp_path / "mosaic.zarr", ds)

    assert pack_store(store, ["red"]).exists()


def test_pack_store_removes_source_and_is_idempotent(tmp_path: Path, dataset: xr.Dataset) -> None:
    store = _store(tmp_path / "mosaic.zarr", dataset)

    dest = pack_store(store, ["red", "green"], remove_source=True)
    mtime = dest.stat().st_mtime_ns

    assert not store.exists()
    assert pack_store(store, ["red", "green"]) == dest
    assert dest.stat().st_mtime_ns == mtime


def test_pack_store_failure_leaves_no_partial_zip(
    tmp_path: Path, dataset: xr.Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path / "mosaic.zarr", dataset)

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("zipfile.ZipFile.write", fail)
    with pytest.raises(OSError, match="disk full"):
        pack_store(store, ["red", "green"])

    assert sorted(p.name for p in tmp_path.iterdir()) == ["mosaic.zarr"]


def test_remove_cache_removes_everything_once_stores_complete(
    tmp_path: Path, dataset: xr.Dataset
) -> None:
    cache = _cache(tmp_path / "cache")
    cached_items = tmp_path / "cached_items.json"
    cached_items.write_text("{}")
    packed = pack_store(_store(tmp_path / "a.zarr", dataset), ["red", "green"], remove_source=True)
    unpacked = _store(tmp_path / "b.zarr", dataset)

    remove_cache(
        cache,
        [packed.with_suffix(""), unpacked],
        ["red", "green"],
        cached_items_path=cached_items,
    )

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
        remove_cache(cache, stores, ["red", "green"], cached_items_path=cached_items)

    assert (cache / "item-a" / "red.tif").exists()
    assert cached_items.exists()


def test_remove_cache_refuses_without_stores(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "cache")

    with pytest.raises(ValueError, match="No stores"):
        remove_cache(cache, [], ["red"])

    assert cache.exists()
