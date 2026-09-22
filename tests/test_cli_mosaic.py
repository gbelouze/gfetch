import datetime
from pathlib import Path

import numpy as np
import pystac
import xarray as xr
import yaml
from odc.geo.geobox import GeoBox

from gfetch.cli.config import Config, load
from gfetch.cli.mosaic import mosaic as mosaic_cmd
from gfetch.mosaic import group_by_utm_zone
from gfetch.write import region_is_written

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
        "patch_chunks": 2,  # patch = 8x8 pixels = 2x2 native chunks -> 2x2 = 4 patches
        **overrides,
    }
    path.write_text(yaml.dump(config_dict))
    return path


def _write_fake_items(cfg: Config) -> None:
    # A null geometry doesn't survive a save/reload round-trip with its bbox intact
    # (pystac drops bbox along with it), so this needs a real matching polygon -
    # unlike test_mosaic.py's in-memory-only `_item` helper, this item is read back
    # from disk exactly like `gfetch mosaic` reads real ones.
    left, bottom, right, top = cfg.aoi.bbox
    corners = [[left, bottom], [right, bottom], [right, top], [left, top], [left, bottom]]
    geometry = {"type": "Polygon", "coordinates": [corners]}
    item = pystac.Item(
        id="fake",
        geometry=geometry,
        bbox=list(cfg.aoi.bbox),
        datetime=datetime.datetime(2024, 3, 1, tzinfo=datetime.UTC),
        properties={},
    )
    pystac.ItemCollection([item]).save_object(str(cfg.cached_items_path))


def _fake_zone_geobox(crs: object, aoi_bbox: object, resolution: object) -> GeoBox:
    return _FIXED_GEOBOX


def _fake_build_mosaic(calls: list[GeoBox]) -> object:
    def build(zone_items: list[pystac.Item], geobox: GeoBox, bands: list[str], **kwargs: object):
        calls.append(geobox)
        data = {
            band: (("y", "x"), np.zeros((geobox.shape.y, geobox.shape.x), dtype="float32"))
            for band in bands
        }
        # Must stay dask-backed like the real `mosaic()`'s output - otherwise
        # `prepare_template`'s `compute=False` has nothing to defer and writes real
        # (all-zero) chunk data immediately, making every patch look already-written.
        return xr.Dataset(data).chunk({"y": 4, "x": 4})

    return build


def test_mosaic_computes_every_patch_on_a_fresh_store(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _write_config(tmp_path / "config.yaml")
    cfg = load(cfg_path)
    _write_fake_items(cfg)

    calls: list[GeoBox] = []
    monkeypatch.setattr("gfetch.cli.mosaic.zone_geobox", _fake_zone_geobox)
    monkeypatch.setattr("gfetch.cli.mosaic.build_mosaic", _fake_build_mosaic(calls))

    mosaic_cmd(cfg_path)

    # 1 call to build the lazy template + 4 patches (2x2 native chunks per patch over
    # a 16x16 grid with 4x4 native chunks).
    assert len(calls) == 5


def test_mosaic_resumes_by_skipping_already_written_patches(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _write_config(tmp_path / "config.yaml")
    cfg = load(cfg_path)
    _write_fake_items(cfg)

    monkeypatch.setattr("gfetch.cli.mosaic.zone_geobox", _fake_zone_geobox)

    calls: list[GeoBox] = []
    monkeypatch.setattr("gfetch.cli.mosaic.build_mosaic", _fake_build_mosaic(calls))
    mosaic_cmd(cfg_path)
    assert len(calls) == 5

    calls.clear()
    mosaic_cmd(cfg_path)
    # Store already initialized (skipped before ever building the template dataset)
    # and every patch already written - nothing left to call build_mosaic for at all.
    assert len(calls) == 0


def test_mosaic_splits_disjoint_patches_across_tasks(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _write_config(tmp_path / "config.yaml")
    cfg = load(cfg_path)
    _write_fake_items(cfg)

    monkeypatch.setattr("gfetch.cli.mosaic.zone_geobox", _fake_zone_geobox)

    calls_0: list[GeoBox] = []
    monkeypatch.setattr("gfetch.cli.mosaic.build_mosaic", _fake_build_mosaic(calls_0))
    mosaic_cmd(cfg_path, task_id=0, n_tasks=2)
    assert len(calls_0) == 1 + 2  # template + this task's half of the 4 patches

    calls_1: list[GeoBox] = []
    monkeypatch.setattr("gfetch.cli.mosaic.build_mosaic", _fake_build_mosaic(calls_1))
    mosaic_cmd(cfg_path, task_id=1, n_tasks=2)
    # Store already initialized by task 0 - just this task's half of the 4 patches,
    # disjoint from task 0's, nothing skipped.
    assert len(calls_1) == 2

    items = list(pystac.ItemCollection.from_file(cfg.cached_items_path))
    (crs,) = group_by_utm_zone(items, cfg.aoi.bbox)
    path = cfg.zarr_path(crs)
    assert region_is_written(path, {"y": slice(0, 16), "x": slice(0, 16)}, ["red"])
