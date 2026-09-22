from pathlib import Path

import yaml
from odc.geo.crs import CRS

from gfetch.cli.config import load


def _write_config(path: Path, **overrides: object) -> Path:
    config_dict = {
        "aoi": {"left": 2.2, "bottom": 48.7, "right": 2.5, "top": 49.0},
        "time_range": {"start": "2024-01-01", "end": "2024-06-01"},
        "output_dir": str(path.parent),
        **overrides,
    }
    path.write_text(yaml.dump(config_dict))
    return path


def test_load_minimal_config(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml")
    cfg = load(config_path)

    assert cfg.aoi.bbox == (2.2, 48.7, 2.5, 49.0)
    assert cfg.time_range.datetime == "2024-01-01/2024-06-01"
    assert cfg.satellite == "sentinel-2"
    assert cfg.source == "earthsearch"
    assert cfg.bands is None
    assert cfg.resampling == {}
    assert cfg.orbit_state is None
    assert cfg.output_dir == Path(tmp_path).expanduser().absolute()
    assert cfg.patch_chunks == 1


def test_config_derived_paths(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml")
    cfg = load(config_path)

    assert cfg.cache_dir == cfg.output_dir / "cache"
    assert cfg.items_path == cfg.output_dir / "items.json"
    assert cfg.cached_items_path == cfg.output_dir / "cached_items.json"
    assert cfg.zarr_path(CRS("EPSG:32736")) == cfg.output_dir / "mosaic_epsg32736.zarr"


def test_load_overrides(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        satellite="sentinel-2",
        source="planetary-computer",
        bands=["red", "green"],
        max_cloud_cover=20.0,
        n_workers=8,
        resampling={"*": "bilinear", "scl": "nearest"},
        orbit_state="ascending",
        patch_chunks=10,
    )
    cfg = load(config_path)

    assert cfg.source == "planetary-computer"
    assert cfg.bands == ["red", "green"]
    assert cfg.max_cloud_cover == 20.0
    assert cfg.n_workers == 8
    assert cfg.resampling == {"*": "bilinear", "scl": "nearest"}
    assert cfg.orbit_state == "ascending"
    assert cfg.patch_chunks == 10
