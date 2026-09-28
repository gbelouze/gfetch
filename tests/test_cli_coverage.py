import datetime
from pathlib import Path

import geopandas as gpd
import pystac
import pytest
import shapely
import yaml
from odc.geo.geobox import GeoBox

from gfetch import countries as countries_module
from gfetch.cli.config import load
from gfetch.cli.coverage import coverage as coverage_cmd

# Same fixed grid and country polygon as `test_cli_mosaic.py`'s shard-skipping test,
# so the shards flagged here are the ones `mosaic` records as skipped.
_FIXED_GEOBOX = GeoBox.from_bbox(
    (245440, 6224940, 245456, 6224956), crs="EPSG:3857", resolution=1.0
)
_POLYGON = shapely.box(2.20483, 48.70492, 2.20486, 48.70494)


def _write_job(tmp_path: Path, **overrides: object) -> Path:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.dump(
            {
                "countries": ["Somewhere"],
                "time_range": {"start": "2024-01-01", "end": "2024-06-01"},
                "output_dir": str(tmp_path),
                "bands": ["red"],
                "chunks": {"x": 4, "y": 4},
                "shard_factor": 2,
                **overrides,
            }
        )
    )
    return cfg_path


def _write_items(cfg_path: Path, satellite_key: str, states: list[str | None]) -> None:
    cfg = load(cfg_path, satellite_key)
    left, bottom, right, top = cfg.resolved_aoi.bbox
    corners = [[left, bottom], [right, bottom], [right, top], [left, top], [left, bottom]]
    items = [
        pystac.Item(
            id=f"item{i}",
            geometry={"type": "Polygon", "coordinates": [corners]},
            bbox=list(cfg.resolved_aoi.bbox),
            datetime=datetime.datetime(2024, 3, 1 + i, tzinfo=datetime.UTC),
            properties={"sat:orbit_state": state} if state is not None else {},
        )
        for i, state in enumerate(states)
    ]
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pystac.ItemCollection(items).save_object(str(cfg.items_path))


@pytest.fixture(autouse=True)
def _fixed_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(countries_module, "resolve_country_polygon", lambda names: _POLYGON)
    monkeypatch.setattr("gfetch.cli.coverage.zone_geobox", lambda *args: _FIXED_GEOBOX)


def test_coverage_writes_one_row_per_shard(tmp_path: Path) -> None:
    cfg_path = _write_job(tmp_path)
    _write_items(cfg_path, "s2", [None, None])

    coverage_cmd(cfg_path, "s2")

    gdf = gpd.read_parquet(load(cfg_path, "s2").coverage_path)
    assert gdf.crs == "EPSG:4326"
    # The items only cover the AOI's bbox, within the one shard not skipped.
    columns = ["y", "x", "skipped", "n_items", "n_timesteps"]
    assert sorted(gdf[columns].itertuples(index=False, name=None)) == [
        (0, 0, True, 0, 0),
        (0, 8, True, 0, 0),
        (8, 0, False, 2, 2),
        (8, 8, True, 0, 0),
    ]
    assert "n_ascending" not in gdf.columns


def test_coverage_counts_orbit_states_when_split_into_bands(tmp_path: Path) -> None:
    cfg_path = _write_job(tmp_path, orbit_state="as_bands", bands=["vv"])
    _write_items(cfg_path, "s1", ["ascending", "ascending", "descending"])

    coverage_cmd(cfg_path, "s1")

    gdf = gpd.read_parquet(load(cfg_path, "s1").coverage_path)
    shard = gdf[~gdf["skipped"]]
    assert shard[["n_ascending", "n_descending"]].values.tolist() == [[2, 1]]


def test_coverage_needs_search_results(tmp_path: Path) -> None:
    cfg_path = _write_job(tmp_path)

    coverage_cmd(cfg_path, "s2")

    assert not load(cfg_path, "s2").coverage_path.exists()
