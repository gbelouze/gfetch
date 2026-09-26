import datetime
from collections.abc import Callable
from pathlib import Path
from typing import cast

import dask
import numpy as np
import pystac
import xarray as xr
import yaml
import zarr
from odc.geo.geobox import GeoBox

from gfetch.cli.config import Config, load
from gfetch.cli.mosaic import mosaic as mosaic_cmd
from gfetch.finalize import pack_store
from gfetch.mosaic import group_by_utm_zone
from gfetch.write import region_is_written, store_is_complete

# Fixed regardless of the real AOI/CRS, so every test gets an exact, hand-picked
# pixel grid instead of depending on real UTM reprojection arithmetic.
_FIXED_GEOBOX = GeoBox.from_bbox((0, 0, 16, 16), crs="EPSG:3857", resolution=1.0)


def _write_config(path: Path, **overrides: object) -> Path:
    config_dict = {
        "aoi": {"left": 2.2, "bottom": 48.7, "right": 2.21, "top": 48.71},
        "time_range": {"start": "2024-01-01", "end": "2024-06-01"},
        "output_dir": str(path.parent),
        "bands": ["red"],
        "chunks": {"x": 4, "y": 4},
        "shard_factor": 2,  # 2x2 chunks per shard -> 2x2 = 4 shards
        "compute_chunk_factor": 1,
        **overrides,
    }
    path.write_text(yaml.dump(config_dict))
    return path


def _write_fake_items(cfg: Config) -> None:
    # A null geometry doesn't survive a save/reload round-trip with its bbox intact
    # (pystac drops bbox along with it), so this needs a real matching polygon -
    # unlike test_mosaic.py's in-memory-only `_item` helper, this item is read back
    # from disk exactly like `gfetch mosaic` reads real ones.
    left, bottom, right, top = cfg.resolved_aoi.bbox
    corners = [[left, bottom], [right, bottom], [right, top], [left, top], [left, bottom]]
    geometry = {"type": "Polygon", "coordinates": [corners]}
    item = pystac.Item(
        id="fake",
        geometry=geometry,
        bbox=list(cfg.resolved_aoi.bbox),
        datetime=datetime.datetime(2024, 3, 1, tzinfo=datetime.UTC),
        properties={},
    )
    pystac.ItemCollection([item]).save_object(str(cfg.cached_items_path))


def _fake_zone_geobox(crs: object, aoi_bbox: object, resolution: object) -> GeoBox:
    return _FIXED_GEOBOX


def _fake_build_mosaic(calls: list[GeoBox]) -> Callable[..., xr.Dataset]:
    def build(
        zone_items: list[pystac.Item],
        geobox: GeoBox,
        bands: list[str],
        on_load: Callable[[xr.Dataset], object] | None = None,
        **kwargs: object,
    ):
        calls.append(geobox)
        data = {
            band: (("y", "x"), np.zeros((geobox.shape.y, geobox.shape.x), dtype="float32"))
            for band in bands
        }
        # Must stay dask-backed like the real `mosaic()`'s output - otherwise
        # `prepare_template`'s `compute=False` has nothing to defer and writes real
        # (all-zero) chunk data immediately, making every patch look already-written.
        chunks = cast("dict[str, int] | None", kwargs.get("chunks")) or {"y": 4, "x": 4}
        ds = xr.Dataset(data).chunk({"y": chunks["y"], "x": chunks["x"]})
        if on_load is not None:
            on_load(ds)
        return ds

    return build


def test_mosaic_computes_every_patch_on_a_fresh_store(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _write_config(tmp_path / "config.yaml")
    cfg = load(cfg_path, "s2")
    _write_fake_items(cfg)

    calls: list[GeoBox] = []
    monkeypatch.setattr("gfetch.cli.mosaic.zone_geobox", _fake_zone_geobox)
    monkeypatch.setattr("gfetch.cli.mosaic.build_mosaic", _fake_build_mosaic(calls))

    mosaic_cmd(cfg_path, "s2")

    # 1 call to build the lazy template + 4 patches (2x2 native chunks per patch over
    # a 16x16 grid with 4x4 native chunks).
    assert len(calls) == 5


def test_mosaic_builds_the_template_at_shard_size(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _write_config(tmp_path / "config.yaml")
    cfg = load(cfg_path, "s2")
    _write_fake_items(cfg)

    calls: list[GeoBox] = []
    chunks_seen: list[dict[str, int]] = []
    build = _fake_build_mosaic(calls)

    def spy(*args: object, **kwargs: object) -> xr.Dataset:
        chunks_seen.append(cast("dict[str, int]", kwargs["chunks"]))
        return build(*args, **kwargs)

    monkeypatch.setattr("gfetch.cli.mosaic.zone_geobox", _fake_zone_geobox)
    monkeypatch.setattr("gfetch.cli.mosaic.build_mosaic", spy)

    mosaic_cmd(cfg_path, "s2")

    template, *patches = chunks_seen
    assert (template["y"], template["x"]) == (8, 8)
    assert all((patch["y"], patch["x"]) == (4, 4) for patch in patches)
    items = list(pystac.ItemCollection.from_file(cfg.cached_items_path))
    (crs,) = group_by_utm_zone(items, cfg.resolved_aoi.bbox)
    za = zarr.open_array(store=cfg.zarr_path(crs) / "red")
    assert za.chunks == (4, 4)
    assert za.shards == (8, 8)


def test_mosaic_resumes_by_skipping_already_written_patches(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _write_config(tmp_path / "config.yaml")
    cfg = load(cfg_path, "s2")
    _write_fake_items(cfg)

    monkeypatch.setattr("gfetch.cli.mosaic.zone_geobox", _fake_zone_geobox)

    calls: list[GeoBox] = []
    monkeypatch.setattr("gfetch.cli.mosaic.build_mosaic", _fake_build_mosaic(calls))
    mosaic_cmd(cfg_path, "s2")
    assert len(calls) == 5

    calls.clear()
    mosaic_cmd(cfg_path, "s2")
    # Store already initialized (skipped before ever building the template dataset)
    # and every patch already written - nothing left to call build_mosaic for at all.
    assert len(calls) == 0


def test_mosaic_splits_disjoint_patches_across_tasks(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _write_config(tmp_path / "config.yaml")
    cfg = load(cfg_path, "s2")
    _write_fake_items(cfg)

    monkeypatch.setattr("gfetch.cli.mosaic.zone_geobox", _fake_zone_geobox)

    calls_0: list[GeoBox] = []
    monkeypatch.setattr("gfetch.cli.mosaic.build_mosaic", _fake_build_mosaic(calls_0))
    mosaic_cmd(cfg_path, "s2", task_id=0, n_tasks=2)
    assert len(calls_0) == 1 + 2  # template + this task's half of the 4 patches

    calls_1: list[GeoBox] = []
    monkeypatch.setattr("gfetch.cli.mosaic.build_mosaic", _fake_build_mosaic(calls_1))
    mosaic_cmd(cfg_path, "s2", task_id=1, n_tasks=2)
    # Store already initialized by task 0 - just this task's half of the 4 patches,
    # disjoint from task 0's, nothing skipped.
    assert len(calls_1) == 2

    items = list(pystac.ItemCollection.from_file(cfg.cached_items_path))
    (crs,) = group_by_utm_zone(items, cfg.resolved_aoi.bbox)
    path = cfg.zarr_path(crs)
    assert region_is_written(path, {"y": slice(0, 16), "x": slice(0, 16)}, ["red"])


def test_mosaic_skips_packed_zone_without_recreating_its_store(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _write_config(tmp_path / "config.yaml")
    cfg = load(cfg_path, "s2")
    _write_fake_items(cfg)

    monkeypatch.setattr("gfetch.cli.mosaic.zone_geobox", _fake_zone_geobox)
    calls: list[GeoBox] = []
    monkeypatch.setattr("gfetch.cli.mosaic.build_mosaic", _fake_build_mosaic(calls))
    mosaic_cmd(cfg_path, "s2")

    items = list(pystac.ItemCollection.from_file(cfg.cached_items_path))
    (crs,) = group_by_utm_zone(items, cfg.resolved_aoi.bbox)
    pack_store(cfg.zarr_path(crs), ["red"], remove_source=True)

    calls.clear()
    mosaic_cmd(cfg_path, "s2")

    assert calls == []
    assert not cfg.zarr_path(crs).exists()


def test_mosaic_tunes_worker_count_within_range(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _write_config(tmp_path / "config.yaml", n_compute_workers=[1, 3])
    cfg = load(cfg_path, "s2")
    _write_fake_items(cfg)
    monkeypatch.setattr("gfetch.cli.mosaic.zone_geobox", _fake_zone_geobox)

    workers_seen: list[int | None] = []
    fake_build = _fake_build_mosaic([])

    def recording_build(zone_items, geobox, bands, **kwargs):
        workers_seen.append(dask.config.get("num_workers", None))
        return fake_build(zone_items, geobox, bands, **kwargs)

    monkeypatch.setattr("gfetch.cli.mosaic.build_mosaic", recording_build)
    mosaic_cmd(cfg_path, "s2")

    # Template build first (outside any shard), then min, max, middle, then a tuned pick.
    assert workers_seen[1:4] == [1, 3, 2]
    assert workers_seen[4] in {1, 2, 3}


def test_mosaic_computes_in_bricks_of_several_store_chunks(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _write_config(tmp_path / "config.yaml", compute_chunk_factor=2)
    cfg = load(cfg_path, "s2")
    _write_fake_items(cfg)
    monkeypatch.setattr("gfetch.cli.mosaic.zone_geobox", _fake_zone_geobox)

    requested_chunks: list[dict] = []
    fake_build = _fake_build_mosaic([])

    def recording_build(zone_items, geobox, bands, **kwargs):
        requested_chunks.append(kwargs["chunks"])
        return fake_build(zone_items, geobox, bands, **kwargs)

    monkeypatch.setattr("gfetch.cli.mosaic.build_mosaic", recording_build)
    mosaic_cmd(cfg_path, "s2")

    assert all(c["y"] == 8 and c["x"] == 8 for c in requested_chunks)
    items = list(pystac.ItemCollection.from_file(cfg.cached_items_path))
    (crs,) = group_by_utm_zone(items, cfg.resolved_aoi.bbox)
    za = zarr.open_array(store=cfg.zarr_path(crs) / "red")
    assert za.chunks == (4, 4)
    assert za.shards == (8, 8)
    assert store_is_complete(cfg.zarr_path(crs), ["red"])
