import datetime
import shutil
from pathlib import Path

import numpy as np
import pystac
import pytest
import xarray as xr
import yaml
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from gfetch.cli.config import Config, load
from gfetch.cli.finalize import clean as clean_cmd
from gfetch.cli.finalize import vrt as vrt_cmd
from gfetch.mosaic import group_by_utm_zone
from gfetch.vrt import vrt_path
from gfetch.write import prepare_template, write


def _setup_job(tmp_path: Path, *, complete: bool = True) -> tuple[Path, Config, Path]:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.dump(
            {
                "aoi": {"left": 2.2, "bottom": 48.7, "right": 2.21, "top": 48.71},
                "time_range": {"start": "2024-01-01", "end": "2024-06-01"},
                "output_dir": str(tmp_path),
                "bands": ["red"],
            }
        )
    )
    cfg = load(cfg_path, "s2")
    cfg.output_dir.mkdir(parents=True)
    left, bottom, right, top = cfg.resolved_aoi.bbox
    corners = [[left, bottom], [right, bottom], [right, top], [left, top], [left, bottom]]
    item = pystac.Item(
        id="fake",
        geometry={"type": "Polygon", "coordinates": [corners]},
        bbox=list(cfg.resolved_aoi.bbox),
        datetime=datetime.datetime(2024, 3, 1, tzinfo=datetime.UTC),
        properties={},
    )
    pystac.ItemCollection([item]).save_object(str(cfg.items_path))
    pystac.ItemCollection([item]).save_object(str(cfg.cached_items_path))
    (cfg.cache_dir / "fake").mkdir(parents=True)
    (cfg.cache_dir / "fake" / "red.tif").write_bytes(b"x")

    (crs,) = group_by_utm_zone([item], cfg.resolved_aoi.bbox)
    store = cfg.zarr_path(crs)
    ds = xr.Dataset(
        {"red": (("y", "x"), np.ones((8, 8), dtype="float32"))},
        coords=xr_coords(GeoBox.from_bbox((0, 0, 80, 80), crs=crs, resolution=10)),
    ).chunk({"y": 4, "x": 4})
    if complete:
        write(ds, store)
    else:
        prepare_template(ds, store)
    return cfg_path, cfg, store


def test_vrt_writes_one_per_zone_store(tmp_path: Path) -> None:
    cfg_path, _, store = _setup_job(tmp_path, complete=False)

    vrt_cmd(cfg_path, "s2")

    assert vrt_path(store).exists()


def test_vrt_skips_missing_store(tmp_path: Path) -> None:
    cfg_path, _, store = _setup_job(tmp_path)
    shutil.rmtree(store)

    vrt_cmd(cfg_path, "s2")

    assert not vrt_path(store).exists()


def test_clean_removes_cache_once_stores_complete(tmp_path: Path) -> None:
    cfg_path, cfg, _ = _setup_job(tmp_path)

    clean_cmd(cfg_path, "s2")

    assert not cfg.cache_dir.exists()
    assert not cfg.cached_items_path.exists()
    assert cfg.items_path.exists()


def test_clean_refuses_while_a_store_is_incomplete(tmp_path: Path) -> None:
    cfg_path, cfg, _ = _setup_job(tmp_path, complete=False)

    with pytest.raises(ValueError, match="incomplete"):
        clean_cmd(cfg_path, "s2")

    assert (cfg.cache_dir / "fake" / "red.tif").exists()
    assert cfg.cached_items_path.exists()
