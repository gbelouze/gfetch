import datetime
from collections.abc import Callable
from pathlib import Path
from typing import cast

import dask
import numpy as np
import pystac
import pytest
import shapely
import xarray as xr
import yaml
import zarr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from gfetch import countries as countries_module
from gfetch.cli.config import Config, load
from gfetch.cli.mosaic import mosaic as mosaic_cmd
from gfetch.mosaic import group_by_utm_zone
from gfetch.ocm import ocm_path
from gfetch.write import (
    BAND_NAMES_ATTR,
    SKIPPED_SHARDS_ATTR,
    STACKED_VARIABLE,
    region_is_written,
    store_is_complete,
)

# Fixed regardless of the real CRS, so every test gets an exact, hand-picked pixel grid
# instead of depending on real UTM reprojection arithmetic. It lies inside
# `_write_config`'s AOI, so no shard is skipped as outside it.
_FIXED_GEOBOX = GeoBox.from_bbox(
    (245440, 6224940, 245456, 6224956), crs="EPSG:3857", resolution=1.0
)


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
        ds = xr.Dataset(data, coords=xr_coords(geobox)).chunk({"y": chunks["y"], "x": chunks["x"]})
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
    za = zarr.open_array(store=cfg.zarr_path(crs) / STACKED_VARIABLE)
    assert za.chunks == (1, 4, 4)
    assert za.shards == (1, 8, 8)


def test_mosaic_stacks_bands_into_one_array(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _write_config(tmp_path / "config.yaml", bands=["red", "green"])
    cfg = load(cfg_path, "s2")
    _write_fake_items(cfg)
    monkeypatch.setattr("gfetch.cli.mosaic.zone_geobox", _fake_zone_geobox)
    monkeypatch.setattr("gfetch.cli.mosaic.build_mosaic", _fake_build_mosaic([]))

    mosaic_cmd(cfg_path, "s2")

    items = list(pystac.ItemCollection.from_file(cfg.cached_items_path))
    (crs,) = group_by_utm_zone(items, cfg.resolved_aoi.bbox)
    path = cfg.zarr_path(crs)
    group = zarr.open_group(store=path, mode="r")
    assert sorted(group.array_keys()) == sorted(["band", STACKED_VARIABLE, "spatial_ref", "x", "y"])
    za = group[STACKED_VARIABLE]
    assert isinstance(za, zarr.Array)
    assert za.attrs[BAND_NAMES_ATTR] == ["red", "green"]
    assert (za.chunks, za.shards) == ((2, 4, 4), (2, 8, 8))
    assert store_is_complete(path, [STACKED_VARIABLE])


def test_mosaic_refuses_a_store_with_other_bands(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _write_config(tmp_path / "config.yaml", bands=["red", "green"])
    _write_fake_items(load(cfg_path, "s2"))
    monkeypatch.setattr("gfetch.cli.mosaic.zone_geobox", _fake_zone_geobox)
    monkeypatch.setattr("gfetch.cli.mosaic.build_mosaic", _fake_build_mosaic([]))
    mosaic_cmd(cfg_path, "s2")

    _write_config(tmp_path / "config.yaml", bands=["green", "red"])
    with pytest.raises(ValueError, match="on-disk bands"):
        mosaic_cmd(cfg_path, "s2")


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
    assert region_is_written(path, {"y": slice(0, 16), "x": slice(0, 16)}, [STACKED_VARIABLE])


@pytest.mark.parametrize("from_cache", [True, False])
def test_mosaic_signs_hrefs_only_when_loading_remotely(
    tmp_path: Path, monkeypatch, from_cache: bool
) -> None:
    cfg_path = _write_config(tmp_path / "config.yaml")
    cfg = load(cfg_path, "s2")
    _write_fake_items(cfg)
    if not from_cache:
        cfg.cached_items_path.rename(cfg.items_path)
    monkeypatch.setattr("gfetch.cli.mosaic.zone_geobox", _fake_zone_geobox)
    signers: list[object] = []
    fake_build = _fake_build_mosaic([])

    def recording_build(zone_items, geobox, bands, *, patch_url, **kwargs):
        signers.append(patch_url)
        return fake_build(zone_items, geobox, bands, **kwargs)

    monkeypatch.setattr("gfetch.cli.mosaic.build_mosaic", recording_build)
    mosaic_cmd(cfg_path, "s2")

    # Template build, then one call per shard.
    assert len(signers) == 5
    if from_cache:
        assert signers == [None] * 5
    else:
        assert all(callable(s) for s in signers)
        assert len({id(s) for s in signers}) == 5


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
    za = zarr.open_array(store=cfg.zarr_path(crs) / STACKED_VARIABLE)
    assert za.chunks == (1, 4, 4)
    assert za.shards == (1, 8, 8)
    assert store_is_complete(cfg.zarr_path(crs), [STACKED_VARIABLE])


def test_mosaic_skips_shards_outside_the_country_polygon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Within the fixed geobox's bottom-left 8x8 shard, clear of the others.
    polygon = shapely.box(2.20483, 48.70492, 2.20486, 48.70494)
    monkeypatch.setattr(countries_module, "resolve_country_polygon", lambda names: polygon)
    cfg_path = _write_config(tmp_path / "config.yaml", aoi=None, countries=["Somewhere"])
    cfg = load(cfg_path, "s2")
    _write_fake_items(cfg)

    calls: list[GeoBox] = []
    monkeypatch.setattr("gfetch.cli.mosaic.zone_geobox", _fake_zone_geobox)
    monkeypatch.setattr("gfetch.cli.mosaic.build_mosaic", _fake_build_mosaic(calls))
    mosaic_cmd(cfg_path, "s2")

    assert len(calls) == 1 + 1  # template + the one shard in the AOI
    assert calls[1] == _FIXED_GEOBOX[8:16, 0:8]
    items = list(pystac.ItemCollection.from_file(cfg.cached_items_path))
    (crs,) = group_by_utm_zone(items, cfg.resolved_aoi.bbox)
    path = cfg.zarr_path(crs)
    assert zarr.open_group(store=path, mode="r").attrs[SKIPPED_SHARDS_ATTR] == {
        "dimensions": ["y", "x"],
        "indices": [[0, 0], [0, 1], [1, 1]],
    }
    assert store_is_complete(path, [STACKED_VARIABLE])


def test_mosaic_with_ocm_refuses_until_every_mask_exists(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _write_config(tmp_path / "config.yaml", ocm=True, bands=["red", "green", "nir"])
    cfg = load(cfg_path, "s2")
    _write_fake_items(cfg)
    monkeypatch.setattr("gfetch.cli.mosaic.zone_geobox", _fake_zone_geobox)
    seen: list[tuple[list[pystac.Item], object]] = []
    fake_build = _fake_build_mosaic([])

    def recording_build(zone_items, geobox, bands, *, mask_band, **kwargs):
        seen.append((zone_items, mask_band))
        return fake_build(zone_items, geobox, bands, **kwargs)

    monkeypatch.setattr("gfetch.cli.mosaic.build_mosaic", recording_build)

    mosaic_cmd(cfg_path, "s2")
    assert seen == []

    mask = ocm_path(cfg.cache_dir, "fake")
    mask.parent.mkdir(parents=True)
    mask.touch()
    mosaic_cmd(cfg_path, "s2")

    assert seen
    for zone_items, mask_band in seen:
        assert mask_band == "ocm"
        assert [item.assets["ocm"].href for item in zone_items] == [str(mask)]
