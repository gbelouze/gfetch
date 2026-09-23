from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import yaml

from gfetch.cli import gedi as gedi_module
from gfetch.cli.gedi import gedi as gedi_cmd
from gfetch.gedi import GEDI_L2A_DEFAULT_FIELDS


def _write_config(path: Path, **overrides: object) -> Path:
    config_dict = {
        "aoi": {"left": 2.2, "bottom": 48.7, "right": 2.5, "top": 49.0},
        "output": str(path.parent / "l2a.parquet"),
        **overrides,
    }
    path.write_text(yaml.dump(config_dict))
    return path


def _fake_gdf(anc_fields: list[str] | None) -> gpd.GeoDataFrame:
    data: dict[str, object] = {"elevation_lm": [1.0], "elevation_hr": [2.0]}
    for field in anc_fields or []:
        data[field] = [np.arange(101, dtype=float)] if field == "rh" else [1]
    return gpd.GeoDataFrame(data, geometry=gpd.points_from_xy([2.3], [48.8]), crs="EPSG:7912")


def test_gedi_cmd_resolves_rh_into_named_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_fetch(bbox, time_range=None, fields=None, anc_fields=None, progress=None):
        captured["bbox"] = bbox
        captured["time_range"] = time_range
        captured["fields"] = fields
        captured["anc_fields"] = anc_fields
        return _fake_gdf(anc_fields)

    monkeypatch.setattr(gedi_module, "fetch_gedi_l2a", fake_fetch)

    config_path = _write_config(
        tmp_path / "config.yaml",
        time_range={"start": "2020-01-01", "end": "2020-06-01"},
        anc_fields=["quality_flag"],
        rh_percentiles=[0, 50, 100],
    )
    gedi_cmd(config_path)

    output = tmp_path / "l2a.parquet"
    assert output.exists()
    assert captured["bbox"] == (2.2, 48.7, 2.5, 49.0)
    assert captured["time_range"] == ("2020-01-01T00:00:00Z", "2020-06-01T23:59:59Z")
    assert captured["fields"] == GEDI_L2A_DEFAULT_FIELDS
    assert captured["anc_fields"] == ["quality_flag", "rh"]

    gdf = gpd.read_parquet(output)
    assert list(gdf[["rh0", "rh50", "rh100"]].iloc[0]) == [0.0, 50.0, 100.0]
    assert "rh" not in gdf.columns
    assert "quality_flag" in gdf.columns


def test_gedi_cmd_defaults_to_no_time_range_or_anc_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_fetch(bbox, time_range=None, fields=None, anc_fields=None, progress=None):
        captured["time_range"] = time_range
        captured["anc_fields"] = anc_fields
        return _fake_gdf(anc_fields)

    monkeypatch.setattr(gedi_module, "fetch_gedi_l2a", fake_fetch)

    config_path = _write_config(tmp_path / "config.yaml")
    gedi_cmd(config_path)

    assert captured["time_range"] is None
    assert captured["anc_fields"] is None
