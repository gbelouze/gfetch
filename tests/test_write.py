from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from gfetch.write import (
    prepare_template,
    region_is_written,
    store_initialized,
    store_is_complete,
    validate_chunks,
    write,
    write_region,
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

    reopened = xr.open_zarr(path)
    assert np.array_equal(reopened["red"].values, dataset["red"].values)


def test_disjoint_region_write_matches_full_write(tmp_path: Path, dataset: xr.Dataset) -> None:
    template_path = tmp_path / "regions.zarr"
    chunked = dataset.chunk({"x": 8, "y": 20})
    prepare_template(chunked, template_path)

    mid = dataset.sizes["x"] // 2
    for sl in (slice(0, mid), slice(mid, None)):
        write_region(dataset.isel(x=sl), template_path, {"x": sl, "y": slice(None)})

    reopened = xr.open_zarr(template_path)
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

    reopened = xr.open_zarr(path)
    assert np.array_equal(reopened["red"].values, dataset["red"].values)


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
    partial = xr.open_zarr(template_path)
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
