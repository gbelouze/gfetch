from pathlib import Path

import geopandas as gpd
import pytest
import shapely
import yaml

from gfetch import countries as countries_module
from gfetch.cli import gedi as gedi_module
from gfetch.cli.gedi import gedi as gedi_cmd
from gfetch.gedi import GEDI_L2A_DEFAULT_FIELDS, GEDI_L4A_DEFAULT_FIELDS


def _write_config(path: Path, **overrides: object) -> Path:
    config_dict = {
        "aoi": {"left": 2.2, "bottom": 48.7, "right": 2.5, "top": 49.0},
        "output": str(path.parent / "l2a.parquet"),
        **overrides,
    }
    path.write_text(yaml.dump(config_dict))
    return path


def _fake_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"elevation_lm": [1.0], "elevation_hr": [2.0]},
        geometry=gpd.points_from_xy([2.3], [48.8]),
        crs="EPSG:7912",
    )


def test_gedi_cmd_passes_config_to_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_fetch(bbox: tuple, **kwargs: object) -> gpd.GeoDataFrame:
        captured.update(bbox=bbox, **kwargs)
        tile_dir = kwargs["tile_dir"]
        assert isinstance(tile_dir, Path)
        tile_dir.mkdir(parents=True)
        (tile_dir / "tile.parquet").touch()
        return _fake_gdf()

    monkeypatch.setattr(gedi_module, "fetch_gedi_l2a", fake_fetch)

    config_path = _write_config(
        tmp_path / "config.yaml",
        time_range={"start": "2020-01-01", "end": "2020-06-01"},
        anc_fields=["quality_flag"],
        rh_percentiles=[0, 50, 100],
    )
    gedi_cmd(config_path, "l2a")

    assert (tmp_path / "l2a.parquet").exists()
    assert captured["bbox"] == (2.2, 48.7, 2.5, 49.0)
    assert captured["time_range"] == ("2020-01-01T00:00:00Z", "2020-06-01T23:59:59Z")
    assert captured["fields"] == GEDI_L2A_DEFAULT_FIELDS
    assert captured["anc_fields"] == ["quality_flag"]
    assert captured["rh_percentiles"] == [0, 50, 100]
    assert captured["quality_filter"] is True
    assert captured["polygon"] is None
    assert captured["tile_dir"] == tmp_path / "l2a.parquet.tiles"
    assert not captured["tile_dir"].exists()


def test_gedi_cmd_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_fetch(bbox: tuple, **kwargs: object) -> gpd.GeoDataFrame:
        captured.update(kwargs)
        return _fake_gdf()

    monkeypatch.setattr(gedi_module, "fetch_gedi_l2a", fake_fetch)

    gedi_cmd(_write_config(tmp_path / "config.yaml"), "l2a")

    assert captured["time_range"] is None
    assert captured["anc_fields"] is None
    assert captured["rh_percentiles"] is None


def test_gedi_cmd_quality_filter_can_be_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_fetch(bbox: tuple, **kwargs: object) -> gpd.GeoDataFrame:
        captured.update(kwargs)
        return _fake_gdf()

    monkeypatch.setattr(gedi_module, "fetch_gedi_l2a", fake_fetch)

    gedi_cmd(_write_config(tmp_path / "config.yaml", quality_filter=False), "l2a")

    assert captured["quality_filter"] is False


def test_gedi_cmd_passes_country_polygon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    polygon = shapely.Polygon([(29.0, -27.0), (41.0, -27.0), (29.0, -1.0)])
    monkeypatch.setattr(countries_module, "resolve_country_polygon", lambda names: polygon)
    captured = {}

    def fake_fetch(bbox: tuple, **kwargs: object) -> gpd.GeoDataFrame:
        captured.update(bbox=bbox, **kwargs)
        return _fake_gdf()

    monkeypatch.setattr(gedi_module, "fetch_gedi_l2a", fake_fetch)

    config_path = _write_config(tmp_path / "config.yaml", aoi=None, countries=["Mozambique"])
    gedi_cmd(config_path, "l2a")

    assert captured["bbox"] == (29.0, -27.0, 41.0, -1.0)
    assert captured["polygon"] is polygon


def test_gedi_cmd_l4a_fetches_l4a(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_fetch(bbox: tuple, **kwargs: object) -> gpd.GeoDataFrame:
        captured.update(kwargs)
        return _fake_gdf()

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("L2A fetched for an L4A job")

    monkeypatch.setattr(gedi_module, "fetch_gedi_l4a", fake_fetch)
    monkeypatch.setattr(gedi_module, "fetch_gedi_l2a", fail)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "aoi": {"left": 2.2, "bottom": 48.7, "right": 2.5, "top": 49.0},
                "output_dir": str(tmp_path),
                "rh_percentiles": [98],
                "gedi_l4a": {"anc_fields": ["agbd_se"]},
            }
        )
    )
    gedi_cmd(config_path, "l4a")

    assert (tmp_path / "gedi" / "l4a.parquet").exists()
    assert captured["fields"] == GEDI_L4A_DEFAULT_FIELDS
    assert captured["anc_fields"] == ["agbd_se"]
    assert captured["quality_filter"] is True
    assert "rh_percentiles" not in captured
