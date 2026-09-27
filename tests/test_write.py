from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
import zarr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from gfetch.write import (
    SKIPPED_SHARDS_ATTR,
    prepare_template,
    region_is_written,
    store_initialized,
    store_is_complete,
    validate_chunks,
    validate_geobox,
    write,
    write_region,
    write_regions,
)


@pytest.fixture
def dataset() -> xr.Dataset:
    rng = np.random.default_rng(0)
    data = rng.random((20, 16)).astype("float32")
    return xr.Dataset(
        {"red": (("y", "x"), data)},
        coords={"y": np.arange(20), "x": np.arange(16), "spatial_ref": 0},
    )


@pytest.fixture
def dataset_with_crs() -> xr.Dataset:
    """A dataset whose CRS coordinate is CF-compliant but whose data variable has
    already lost the `.encoding["grid_mapping"]` link, e.g. as `mask_clouds`/
    `composite`'s `.where`/`.median` calls leave it downstream of `odc.stac.load`.
    """
    rng = np.random.default_rng(0)
    data = rng.random((20, 16)).astype("float32")
    ds = xr.Dataset(
        {"red": (("y", "x"), data)},
        coords={"y": np.arange(20), "x": np.arange(16), "spatial_ref": 0},
    )
    ds["spatial_ref"].attrs["crs_wkt"] = 'PROJCRS["WGS 84 / UTM zone 31N", ...]'
    assert not ds["red"].encoding
    return ds


def test_write_round_trips(tmp_path: Path, dataset: xr.Dataset) -> None:
    path = tmp_path / "out.zarr"
    write(dataset, path)

    reopened = xr.open_zarr(path, consolidated=False)
    assert np.array_equal(reopened["red"].values, dataset["red"].values)


def test_disjoint_region_write_matches_full_write(tmp_path: Path, dataset: xr.Dataset) -> None:
    template_path = tmp_path / "regions.zarr"
    chunked = dataset.chunk({"x": 8, "y": 20})
    prepare_template(chunked, template_path)

    mid = dataset.sizes["x"] // 2
    for sl in (slice(0, mid), slice(mid, None)):
        write_region(dataset.isel(x=sl), template_path, {"x": sl, "y": slice(None)})

    reopened = xr.open_zarr(template_path, consolidated=False)
    assert np.array_equal(reopened["red"].values, dataset["red"].values)


def test_store_initialized_before_and_after_prepare_template(
    tmp_path: Path, dataset: xr.Dataset
) -> None:
    path = tmp_path / "regions.zarr"
    assert not store_initialized(path)

    prepare_template(dataset.chunk({"x": 8, "y": 20}), path)
    assert store_initialized(path)


def test_prepare_template_is_idempotent(tmp_path: Path, dataset: xr.Dataset) -> None:
    path = tmp_path / "regions.zarr"
    chunked = dataset.chunk({"x": 8, "y": 20})
    prepare_template(chunked, path)

    write_region(dataset, path, {"x": slice(None), "y": slice(None)})
    prepare_template(chunked, path)  # must not re-truncate an already-written store

    reopened = xr.open_zarr(path, consolidated=False)
    assert np.array_equal(reopened["red"].values, dataset["red"].values)


def test_prepare_template_discards_its_template_when_losing_the_race(
    tmp_path: Path, dataset: xr.Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "regions.zarr"
    prepare_template(dataset.chunk({"x": 8, "y": 20}), path)
    # Simulates a caller that checked before the winner's rename landed.
    monkeypatch.setattr("gfetch.write.store_initialized", lambda _: False)

    prepare_template(dataset.chunk({"x": 4, "y": 20}), path)

    validate_chunks(path, ["red"], {"x": 8, "y": 20})
    assert [p.name for p in tmp_path.iterdir()] == ["regions.zarr"]


def test_validate_chunks_passes_when_matching(tmp_path: Path, dataset: xr.Dataset) -> None:
    path = tmp_path / "regions.zarr"
    prepare_template(dataset.chunk({"x": 8, "y": 20}), path)

    validate_chunks(path, ["red"], {"x": 8, "y": 20})  # must not raise


def test_validate_chunks_raises_on_mismatch(tmp_path: Path, dataset: xr.Dataset) -> None:
    path = tmp_path / "regions.zarr"
    prepare_template(dataset.chunk({"x": 8, "y": 20}), path)

    with pytest.raises(ValueError, match="on-disk chunk size"):
        validate_chunks(path, ["red"], {"x": 4, "y": 20})


def test_region_is_written_false_before_write(tmp_path: Path, dataset: xr.Dataset) -> None:
    path = tmp_path / "regions.zarr"
    chunked = dataset.chunk({"x": 8, "y": 20})
    prepare_template(chunked, path)

    assert not region_is_written(path, {"x": slice(0, 8), "y": slice(0, 20)}, ["red"])


def test_region_is_written_true_only_for_the_written_region(
    tmp_path: Path, dataset: xr.Dataset
) -> None:
    path = tmp_path / "regions.zarr"
    chunked = dataset.chunk({"x": 8, "y": 20})
    prepare_template(chunked, path)

    write_region(dataset.isel(x=slice(0, 8)), path, {"x": slice(0, 8), "y": slice(None)})

    assert region_is_written(path, {"x": slice(0, 8), "y": slice(0, 20)}, ["red"])
    assert not region_is_written(path, {"x": slice(8, 16), "y": slice(0, 20)}, ["red"])


def test_region_is_written_requires_every_variable(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            "red": (("y", "x"), rng.random((20, 16)).astype("float32")),
            "green": (("y", "x"), rng.random((20, 16)).astype("float32")),
        },
        coords={"y": np.arange(20), "x": np.arange(16)},
    )
    path = tmp_path / "regions.zarr"
    chunked = ds.chunk({"x": 8, "y": 20})
    prepare_template(chunked, path)

    # Only "red" written - simulates a write_region call killed partway through.
    write_region(ds[["red"]].isel(x=slice(0, 8)), path, {"x": slice(0, 8), "y": slice(None)})

    region = {"x": slice(0, 8), "y": slice(0, 20)}
    assert region_is_written(path, region, ["red"])
    assert not region_is_written(path, region, ["red", "green"])


def test_write_restores_grid_mapping(tmp_path: Path, dataset_with_crs: xr.Dataset) -> None:
    path = tmp_path / "out.zarr"
    write(dataset_with_crs, path)

    red = zarr.open_array(store=path / "red")
    assert red.attrs["grid_mapping"] == "spatial_ref"


def test_disjoint_region_write_restores_grid_mapping(
    tmp_path: Path, dataset_with_crs: xr.Dataset
) -> None:
    path = tmp_path / "regions.zarr"
    chunked = dataset_with_crs.chunk({"x": 8, "y": 20})
    prepare_template(chunked, path)

    mid = dataset_with_crs.sizes["x"] // 2
    for sl in (slice(0, mid), slice(mid, None)):
        write_region(dataset_with_crs.isel(x=sl), path, {"x": sl, "y": slice(None)})

    red = zarr.open_array(store=path / "red")
    assert red.attrs["grid_mapping"] == "spatial_ref"


def test_sharded_template_restores_grid_mapping(
    tmp_path: Path, dataset_with_crs: xr.Dataset
) -> None:
    path = tmp_path / "sharded.zarr"
    prepare_template(dataset_with_crs.chunk({"x": 4, "y": 20}), path, shards={"x": 8})

    red = zarr.open_array(store=path / "red")
    assert red.attrs["grid_mapping"] == "spatial_ref"


def test_write_does_not_consolidate_metadata(tmp_path: Path, dataset_with_crs: xr.Dataset) -> None:
    path = tmp_path / "out.zarr"
    write(dataset_with_crs, path)

    assert zarr.open_group(store=path, mode="r").metadata.consolidated_metadata is None


def test_region_writes_do_not_consolidate_metadata(
    tmp_path: Path, dataset_with_crs: xr.Dataset
) -> None:
    path = tmp_path / "regions.zarr"
    prepare_template(dataset_with_crs.chunk({"x": 4, "y": 20}), path, shards={"x": 8})
    write_region(dataset_with_crs.isel(x=slice(0, 8)), path, {"x": slice(0, 8), "y": slice(None)})

    assert zarr.open_group(store=path, mode="r").metadata.consolidated_metadata is None


def test_disjoint_region_write_workers_do_not_overlap(tmp_path: Path, dataset: xr.Dataset) -> None:
    """Each worker's region write must only touch its own slice - two workers writing
    non-overlapping regions must not corrupt each other's data.
    """
    template_path = tmp_path / "regions.zarr"
    chunked = dataset.chunk({"x": 8, "y": 20})
    prepare_template(chunked, template_path)

    mid = dataset.sizes["x"] // 2
    write_region(
        dataset.isel(x=slice(0, mid)), template_path, {"x": slice(0, mid), "y": slice(None)}
    )

    # Only the first half has been written; the rest of the store should still hold
    # whatever `prepare_template` initialized it to (zeros/fill-value), not garbage
    # and not the first half's data repeated.
    partial = xr.open_zarr(template_path, consolidated=False)
    assert np.array_equal(partial["red"].values[:, :mid], dataset["red"].values[:, :mid])
    assert not np.array_equal(partial["red"].values[:, mid:], dataset["red"].values[:, mid:])


def test_region_is_written_true_for_all_nodata_region(tmp_path: Path) -> None:
    ds = xr.Dataset(
        {"red": (("y", "x"), np.full((20, 16), np.nan, dtype="float32"))},
        coords={"y": np.arange(20), "x": np.arange(16)},
    )
    path = tmp_path / "regions.zarr"
    prepare_template(ds.chunk({"x": 8, "y": 20}), path)

    write_region(ds.isel(x=slice(0, 8)), path, {"x": slice(0, 8), "y": slice(None)})

    assert region_is_written(path, {"x": slice(0, 8), "y": slice(0, 20)}, ["red"])


def test_store_is_complete_only_once_every_chunk_is_written(
    tmp_path: Path, dataset: xr.Dataset
) -> None:
    path = tmp_path / "regions.zarr"
    assert not store_is_complete(path, ["red"])

    prepare_template(dataset.chunk({"x": 8, "y": 20}), path)
    write_region(dataset.isel(x=slice(0, 8)), path, {"x": slice(0, 8), "y": slice(None)})
    assert not store_is_complete(path, ["red"])

    write_region(dataset.isel(x=slice(8, 16)), path, {"x": slice(8, 16), "y": slice(None)})
    assert store_is_complete(path, ["red"])


def test_write_regions_follow_shards_with_clipped_edges(tmp_path: Path) -> None:
    ds = xr.Dataset(
        {"red": (("y", "x"), np.zeros((20, 12), dtype="float32"))},
        coords={"y": np.arange(20), "x": np.arange(12)},
    )
    path = tmp_path / "sharded.zarr"
    prepare_template(ds.chunk({"y": 4, "x": 4}), path, shards={"y": 8, "x": 8})

    za = zarr.open_array(store=path / "red")
    assert za.chunks == (4, 4)
    assert za.shards == (8, 8)
    assert write_regions(path, "red") == [
        {"y": slice(0, 8), "x": slice(0, 8)},
        {"y": slice(0, 8), "x": slice(8, 12)},
        {"y": slice(8, 16), "x": slice(0, 8)},
        {"y": slice(8, 16), "x": slice(8, 12)},
        {"y": slice(16, 20), "x": slice(0, 8)},
        {"y": slice(16, 20), "x": slice(8, 12)},
    ]


def test_write_regions_fall_back_to_chunks_when_unsharded(
    tmp_path: Path, dataset: xr.Dataset
) -> None:
    path = tmp_path / "plain.zarr"
    prepare_template(dataset.chunk({"x": 8, "y": 20}), path)

    assert write_regions(path, "red") == [
        {"y": slice(0, 20), "x": slice(0, 8)},
        {"y": slice(0, 20), "x": slice(8, 16)},
    ]


def test_sharded_region_writes_are_tracked_per_shard(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {"red": (("y", "x"), rng.random((16, 16)).astype("float32"))},
        coords={"y": np.arange(16), "x": np.arange(16)},
    )
    path = tmp_path / "sharded.zarr"
    prepare_template(ds.chunk({"y": 4, "x": 4}), path, shards={"y": 8, "x": 8})
    first, second, *_ = write_regions(path, "red")

    # Dask chunks smaller than the shard, as `mosaic` computes them.
    write_region(ds.isel(first).chunk({"y": 4, "x": 4}), path, first)

    assert region_is_written(path, first, ["red"])
    assert not region_is_written(path, second, ["red"])
    assert [p.relative_to(path).as_posix() for p in path.glob("red/c/*/*")] == ["red/c/0/0"]
    for region in write_regions(path, "red")[1:]:
        write_region(ds.isel(region), path, region)
    assert store_is_complete(path, ["red"])
    xr.testing.assert_equal(xr.open_zarr(path, consolidated=False)["red"].load(), ds["red"])


def test_skipped_shards_are_recorded_and_excluded(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {"red": (("y", "x"), rng.random((16, 16)).astype("float32"))},
        coords={"y": np.arange(16), "x": np.arange(16)},
    )
    path = tmp_path / "sharded.zarr"
    prepare_template(
        ds.chunk({"y": 4, "x": 4}),
        path,
        shards={"y": 8, "x": 8},
        skip=lambda region: region["x"].start == 8,
    )

    attrs = zarr.open_group(store=path, mode="r").attrs[SKIPPED_SHARDS_ATTR]
    assert attrs == {"dimensions": ["y", "x"], "indices": [[0, 1], [1, 1]]}
    regions = write_regions(path, "red")
    assert [(r["y"].start, r["x"].start) for r in regions] == [(0, 0), (8, 0)]
    assert region_is_written(path, {"y": slice(0, 8), "x": slice(8, 16)}, ["red"])

    write_region(ds.isel(regions[0]), path, regions[0])
    assert not store_is_complete(path, ["red"])
    write_region(ds.isel(regions[1]), path, regions[1])
    assert store_is_complete(path, ["red"])
    assert sorted(p.relative_to(path).as_posix() for p in path.glob("red/c/*/*")) == [
        "red/c/0/0",
        "red/c/1/0",
    ]
    reopened = xr.open_zarr(path, consolidated=False)["red"].load()
    assert np.isnan(reopened.isel(x=slice(8, 16))).all()


def test_template_without_skip_records_nothing(tmp_path: Path, dataset: xr.Dataset) -> None:
    path = tmp_path / "sharded.zarr"
    prepare_template(dataset.chunk({"x": 4, "y": 20}), path, shards={"x": 8})

    assert SKIPPED_SHARDS_ATTR not in zarr.open_group(store=path, mode="r").attrs
    assert len(write_regions(path, "red")) == 2


def test_validate_geobox_passes_on_the_store_grid_and_raises_otherwise(tmp_path: Path) -> None:
    geobox = GeoBox.from_bbox((500_000, 9_000_000, 500_160, 9_000_200), "EPSG:32736", resolution=10)
    ds = xr.Dataset(
        {"red": (("y", "x"), np.zeros(geobox.shape, dtype="float32"))},
        coords=xr_coords(geobox),
    )
    path = tmp_path / "out.zarr"
    write(ds, path)

    validate_geobox(path, geobox)
    with pytest.raises(ValueError, match="'x' grid"):
        validate_geobox(
            path,
            GeoBox.from_bbox((499_000, 9_000_000, 500_160, 9_000_200), "EPSG:32736", resolution=10),
        )
    with pytest.raises(ValueError, match="'y' grid"):
        validate_geobox(
            path,
            GeoBox.from_bbox((500_000, 9_000_010, 500_160, 9_000_210), "EPSG:32736", resolution=10),
        )


def test_validate_chunks_raises_on_shard_mismatch(tmp_path: Path, dataset: xr.Dataset) -> None:
    path = tmp_path / "sharded.zarr"
    prepare_template(dataset.chunk({"x": 4, "y": 20}), path, shards={"x": 8})

    validate_chunks(path, ["red"], {"x": 4, "y": 20}, {"x": 8})
    with pytest.raises(ValueError, match="shard size"):
        validate_chunks(path, ["red"], {"x": 4, "y": 20}, {"x": 16})
    with pytest.raises(ValueError, match="shard size"):
        validate_chunks(path, ["red"], {"x": 4, "y": 20})


def test_prepare_template_uses_explicit_chunks_over_dask_chunks(tmp_path: Path) -> None:
    ds = xr.Dataset(
        {"red": (("y", "x"), np.zeros((16, 16), dtype="float32"))},
        coords={"y": np.arange(16), "x": np.arange(16)},
    )
    path = tmp_path / "sharded.zarr"
    prepare_template(
        ds.chunk({"y": 8, "x": 8}), path, shards={"y": 16, "x": 16}, chunks={"y": 4, "x": 4}
    )

    za = zarr.open_array(store=path / "red")
    assert za.chunks == (4, 4)
    assert za.shards == (16, 16)


def test_prepare_template_keeps_a_shard_sized_graph(tmp_path: Path, monkeypatch) -> None:
    ds = xr.Dataset(
        {"red": (("y", "x"), np.zeros((64, 64), dtype="float32"))},
        coords={"y": np.arange(64), "x": np.arange(64)},
    ).chunk({"y": 32, "x": 32})
    written: list[xr.Dataset] = []
    to_zarr = xr.Dataset.to_zarr

    def spy(self: xr.Dataset, *args: Any, **kwargs: Any) -> object:
        written.append(self)
        return to_zarr(self, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, "to_zarr", spy)
    path = tmp_path / "sharded.zarr"
    prepare_template(ds, path, shards={"y": 32, "x": 32}, chunks={"y": 4, "x": 4})

    # A rechunk through the 4x4 store chunks would leave 256 tasks in the graph.
    assert len(dict(written[0]["red"].data.__dask_graph__())) < 16
    za = zarr.open_array(store=path / "red")
    assert za.chunks == (4, 4)
    assert za.shards == (32, 32)
