import json
from pathlib import Path

import pytest
import zarr
from conftest import BANDS, Stores

from gfetch.check import CheckResult, check_store
from gfetch.cli.utils import check
from gfetch.rechunk import rechunk
from gfetch.write import STACKED_VARIABLE


def _failed(results: list[CheckResult]) -> list[str]:
    return [r.name for r in results if not r.ok]


def _edit_json(path: Path, edit: dict) -> None:
    meta = json.loads(path.read_text())
    meta.update(edit)
    path.write_text(json.dumps(meta))


def test_a_store_written_like_mosaic_passes(tmp_path: Path, stores: Stores) -> None:
    path = tmp_path / "store.zarr"
    stores.stacked(path, skip=lambda region: region["x"].start == 16)

    results = check_store(path, 4, 16, BANDS)

    assert _failed(results) == []
    assert [r.name for r in results] == [
        "Zarr v3 group",
        "unconsolidated metadata",
        "stacked bands",
        "band order",
        "chunk and shard size",
        "georeferenced",
        "metadata as mosaic writes it",
        "gfetch:skipped_shards attribute",
        "completely written",
    ]


def test_a_rechunked_per_band_store_passes(tmp_path: Path, stores: Stores) -> None:
    path = tmp_path / "store.zarr"
    stores.old(path, skip=lambda region: region["x"].start == 16)
    assert _failed(check_store(path, 8, 16)) == ["stacked bands"]

    rechunk(path, 8, 16, bands=BANDS)

    assert _failed(check_store(path, 8, 16, BANDS)) == []


def test_another_layout_fails_only_the_layout_check(tmp_path: Path, stores: Stores) -> None:
    path = tmp_path / "store.zarr"
    stores.stacked(path)

    assert _failed(check_store(path, 8, 16)) == ["chunk and shard size"]


def test_another_band_order_fails(tmp_path: Path, stores: Stores) -> None:
    path = tmp_path / "store.zarr"
    stores.stacked(path)

    assert _failed(check_store(path, 4, 16, ["blue", "green", "red"])) == ["band order"]


def test_an_incomplete_store_fails_with_its_unwritten_count(tmp_path: Path, stores: Stores) -> None:
    path = tmp_path / "store.zarr"
    stores.stacked(path)
    next(p for p in (path / STACKED_VARIABLE / "c").rglob("[!.]*") if p.is_file()).unlink()

    results = check_store(path, 4, 16)

    assert _failed(results) == ["completely written"]
    assert results[-1].detail == "1/9 shard(s) unwritten"


def test_consolidated_metadata_fails(tmp_path: Path, stores: Stores) -> None:
    path = tmp_path / "store.zarr"
    stores.stacked(path)
    zarr.consolidate_metadata(path)

    assert "unconsolidated metadata" in _failed(check_store(path, 4, 16))


def test_metadata_drift_fails(tmp_path: Path, stores: Stores) -> None:
    path = tmp_path / "store.zarr"
    stores.stacked(path)
    _edit_json(path / STACKED_VARIABLE / "zarr.json", {"fill_value": 0.0})

    results = check_store(path, 4, 16)

    assert _failed(results) == ["metadata as mosaic writes it"]
    detail = next(r.detail for r in results if r.name == "metadata as mosaic writes it")
    assert "bands/zarr.json: fill_value" in detail


def test_a_store_without_crs_fails(tmp_path: Path, stores: Stores) -> None:
    path = tmp_path / "store.zarr"
    stores.stacked(path)
    meta_path = path / "spatial_ref" / "zarr.json"
    meta = json.loads(meta_path.read_text())
    del meta["attributes"]["crs_wkt"]
    meta_path.write_text(json.dumps(meta))

    assert _failed(check_store(path, 4, 16)) == ["georeferenced"]


def test_a_missing_store_fails(tmp_path: Path) -> None:
    assert _failed(check_store(tmp_path / "missing.zarr", 4, 16)) == ["Zarr v3 group"]


def test_cli_check_reports_whether_every_store_passed(tmp_path: Path, stores: Stores) -> None:
    good, bad = tmp_path / "good.zarr", tmp_path / "bad.zarr"
    stores.stacked(good)
    stores.old(bad)

    assert check([good], chunk=4, shard_factor=4)
    assert not check([good, bad], chunk=4, shard_factor=4)


@pytest.mark.parametrize("argv_tail", [[], ["--bands", "red", "green", "blue"]])
def test_cli_check_exits_with_1_on_failure(
    tmp_path: Path, stores: Stores, argv_tail: list[str]
) -> None:
    from gfetch.cli.main import app

    path = tmp_path / "store.zarr"
    stores.old(path)

    with pytest.raises(SystemExit) as exc:
        app(["utils", "check", str(path), *argv_tail])

    assert exc.value.code == 1
