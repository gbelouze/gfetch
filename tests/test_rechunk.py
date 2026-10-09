import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import xarray as xr
import zarr
from conftest import BANDS, Stores

from gfetch.rechunk import rechunk
from gfetch.write import (
    SKIPPED_SHARDS_ATTR,
    STACKED_VARIABLE,
    prepare_template,
    stack_bands,
    store_is_complete,
)


def _metadata(path: Path) -> dict[str, Any]:
    return {
        str(p.relative_to(path)): json.loads(p.read_text()) for p in sorted(path.rglob("zarr.json"))
    }


def _layout(path: Path) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    arr = zarr.open_array(store=path / STACKED_VARIABLE, mode="r")
    return arr.chunks, arr.shards


def _open(path: Path) -> xr.Dataset:
    return xr.open_zarr(path, consolidated=False).load()


def test_per_band_store_becomes_what_mosaic_writes(tmp_path: Path, stores: Stores) -> None:
    def old_skip(region: dict[str, slice]) -> bool:
        return region["x"].start == 16

    def new_skip(region: dict[str, slice]) -> bool:
        return 16 <= region["x"].start < 32

    order = ["green", "blue", "red"]
    reference = tmp_path / "reference.zarr"
    stores.write(stack_bands(stores.per_band(), order), reference, 8, 8, new_skip)
    path = tmp_path / "store.zarr"
    stores.old(path, old_skip)

    rechunk(path, 8, 8, bands=order)

    assert _metadata(path) == _metadata(reference)
    xr.testing.assert_identical(_open(path), _open(reference))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["reference.zarr", "store.zarr"]


def test_stacked_store_is_rechunked_in_its_band_order(tmp_path: Path, stores: Stores) -> None:
    reference = tmp_path / "reference.zarr"
    stores.write(stack_bands(stores.per_band(), BANDS), reference, 6, 12)
    path = tmp_path / "store.zarr"
    stores.stacked(path, BANDS)

    rechunk(path, 6, 12)

    assert _layout(path) == ((3, 6, 6), (3, 12, 12))
    assert _metadata(path) == _metadata(reference)
    xr.testing.assert_identical(_open(path), _open(reference))


def test_all_nan_shards_are_written(tmp_path: Path, stores: Stores) -> None:
    path = tmp_path / "store.zarr"
    stores.stacked(path, BANDS)

    rechunk(path, 8, 8)

    assert store_is_complete(path, [STACKED_VARIABLE])
    assert np.isnan(_open(path)[STACKED_VARIABLE][:, :8, :8]).all()


def test_skipped_shards_are_recomputed(tmp_path: Path, stores: Stores) -> None:
    path = tmp_path / "store.zarr"
    stores.stacked(path, BANDS, skip=lambda region: region["x"].start == 16)

    rechunk(path, 4, 8)

    attrs = cast(
        "dict[str, list[Any]]", zarr.open_group(store=path, mode="r").attrs[SKIPPED_SHARDS_ATTR]
    )
    assert attrs["dimensions"] == ["y", "x"]
    assert sorted(tuple(i) for i in attrs["indices"]) == [(y, x) for y in range(5) for x in (2, 3)]
    assert store_is_complete(path, [STACKED_VARIABLE])


def test_split_across_tasks_swaps_once_complete(tmp_path: Path, stores: Stores) -> None:
    path = tmp_path / "store.zarr"
    stores.stacked(path, BANDS)
    before = _open(path)

    rechunk(path, 8, 8, task_id=0, n_tasks=2)
    assert _layout(path) == ((3, 4, 4), (3, 16, 16))
    assert (tmp_path / ".store.zarr.rechunk").exists()

    rechunk(path, 8, 8, task_id=1, n_tasks=2)
    assert _layout(path) == ((3, 8, 8), None)
    xr.testing.assert_identical(_open(path), before)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["store.zarr"]


def test_an_interrupted_swap_is_resumed(tmp_path: Path, stores: Stores) -> None:
    path = tmp_path / "store.zarr"
    stores.stacked(path, BANDS)
    before = _open(path)
    rechunk(path, 8, 16, output=tmp_path / ".store.zarr.rechunk")
    path.rename(tmp_path / ".store.zarr.rechunk-old")

    rechunk(path, 8, 16)

    assert _layout(path) == ((3, 8, 8), (3, 16, 16))
    xr.testing.assert_identical(_open(path), before)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["store.zarr"]


def test_output_leaves_the_source_untouched(tmp_path: Path, stores: Stores) -> None:
    path = tmp_path / "store.zarr"
    output = tmp_path / "out.zarr"
    stores.stacked(path, BANDS)

    rechunk(path, 8, 16, output=output)

    assert _layout(path) == ((3, 4, 4), (3, 16, 16))
    assert _layout(output) == ((3, 8, 8), (3, 16, 16))
    xr.testing.assert_identical(_open(output), _open(path))


def test_a_store_in_the_target_layout_is_left_alone(tmp_path: Path, stores: Stores) -> None:
    path = tmp_path / "store.zarr"
    stores.stacked(path, BANDS)

    rechunk(path, 4, 16)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["store.zarr"]


def test_reordering_bands_rewrites_a_store_in_the_target_layout(
    tmp_path: Path, stores: Stores
) -> None:
    path = tmp_path / "store.zarr"
    stores.stacked(path, BANDS)

    rechunk(path, 4, 16, bands=["blue", "green", "red"])

    group = zarr.open_group(store=path, mode="r")
    assert group[STACKED_VARIABLE].attrs["band_names"] == ["blue", "green", "red"]


def test_a_per_band_store_needs_bands(tmp_path: Path, stores: Stores) -> None:
    path = tmp_path / "store.zarr"
    stores.old(path)

    with pytest.raises(ValueError, match="pass the bands"):
        rechunk(path, 8, 16)
    with pytest.raises(ValueError, match="don't match"):
        rechunk(path, 8, 16, bands=["red", "green"])


def test_an_incomplete_store_is_refused(tmp_path: Path, stores: Stores) -> None:
    path = tmp_path / "store.zarr"
    prepare_template(
        stores.per_band().chunk({"y": 16, "x": 16}),
        path,
        shards={"y": 16, "x": 16},
        chunks={"y": 4, "x": 4},
    )

    with pytest.raises(ValueError, match="not completely written"):
        rechunk(path, 8, 16, bands=BANDS)


def test_a_shard_not_multiple_of_the_chunk_is_refused(tmp_path: Path, stores: Stores) -> None:
    with pytest.raises(ValueError, match="not a multiple"):
        rechunk(tmp_path / "store.zarr", 8, 12)
