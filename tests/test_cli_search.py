from pathlib import Path
from typing import Any

import pytest

from gfetch.cli.config import AOIConfig, Config, TimeRangeConfig
from gfetch.cli.search import _build_query


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
