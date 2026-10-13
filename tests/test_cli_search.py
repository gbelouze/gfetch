from pathlib import Path
from typing import Any

import pystac
import pytest
import yaml

from gfetch.cli.config import AOIConfig, Config, TimeRangeConfig, load
from gfetch.cli.search import _build_query
from gfetch.cli.search import search as search_cmd


def _config(**overrides: Any) -> Config:
    return Config(
        aoi=AOIConfig(2.2, 48.7, 2.5, 49.0),
        time_range=TimeRangeConfig("2024-01-01", "2024-06-01"),
        output_dir=Path("/tmp/gfetch-test"),
        **overrides,
    )


def test_build_query_no_filters_returns_none() -> None:
    assert _build_query(_config()) is None


def test_build_query_cloud_cover_only() -> None:
    query = _build_query(_config(max_cloud_cover=20.0))
    assert query == {"eo:cloud_cover": {"lt": 20.0}}


def test_build_query_orbit_state_only() -> None:
    query = _build_query(_config(orbit_state="ascending"))
    assert query == {"sat:orbit_state": {"eq": "ascending"}}


def test_build_query_merges_cloud_cover_and_orbit_state() -> None:
    query = _build_query(_config(max_cloud_cover=20.0, orbit_state="descending"))
    assert query == {
        "eo:cloud_cover": {"lt": 20.0},
        "sat:orbit_state": {"eq": "descending"},
    }


def test_build_query_orbit_state_as_bands_searches_both() -> None:
    assert _build_query(_config(orbit_state="as_bands")) is None


def test_build_query_invalid_orbit_state_raises() -> None:
    with pytest.raises(ValueError, match="orbit_state"):
        _build_query(_config(orbit_state="sideways"))


def _write_config(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.dump(
            {
                "aoi": {"left": 2.2, "bottom": 48.7, "right": 2.21, "top": 48.71},
                "time_range": {"start": "2024-01-01", "end": "2024-06-01"},
                "output_dir": str(tmp_path / "out"),
            }
        )
    )
    return cfg_path


def test_search_writes_items_when_none_exist(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _write_config(tmp_path)
    monkeypatch.setattr("gfetch.cli.search.search_items", lambda *args, **kwargs: [])

    search_cmd(cfg_path, "s2")

    assert load(cfg_path, "s2").items_path.exists()


def test_search_skips_when_items_exist(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _write_config(tmp_path)
    cfg = load(cfg_path, "s2")
    cfg.output_dir.mkdir(parents=True)
    pystac.ItemCollection([]).save_object(str(cfg.items_path))

    def no_search(*args: object, **kwargs: object) -> None:
        raise AssertionError("search must be skipped")

    monkeypatch.setattr("gfetch.cli.search.search_items", no_search)

    search_cmd(cfg_path, "s2")
