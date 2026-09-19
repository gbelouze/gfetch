from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from gfetch.write import prepare_template, write, write_region


@pytest.fixture
def dataset() -> xr.Dataset:
    rng = np.random.default_rng(0)
    data = rng.random((20, 16)).astype("float32")
    return xr.Dataset(
        {"red": (("y", "x"), data)},
        coords={"y": np.arange(20), "x": np.arange(16), "spatial_ref": 0},
    )


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
