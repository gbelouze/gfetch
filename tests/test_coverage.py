import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pystac
import pytest
import xarray as xr
from odc.geo.crs import CRS
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from gfetch.coverage import coverage, shard_regions, solar_day
from gfetch.write import prepare_template, write_regions

CRS_31N = CRS("EPSG:32631")


def _geobox() -> GeoBox:
    # 20x30 px at 10 m: 2x3 shards of 10 px, near 3°E, 0.45°N.
    return GeoBox.from_bbox((500_000, 50_000, 500_300, 50_200), crs=CRS_31N, resolution=10)


def _item(
    item_id: str,
    geobox: GeoBox,
    when: datetime.datetime,
    *,
    properties: dict[str, str] | None = None,
) -> pystac.Item:
    geometry = geobox.extent.to_crs("EPSG:4326").json
    return pystac.Item(
        id=item_id,
        geometry=geometry,
        bbox=list(geobox.extent.to_crs("EPSG:4326").boundingbox),
        datetime=when,
        properties=dict(properties) if properties is not None else {},
    )


def _day(day: int, hour: int = 10) -> datetime.datetime:
    return datetime.datetime(2024, 3, day, hour, tzinfo=datetime.UTC)


@pytest.mark.parametrize("unit", [{"y": 10, "x": 10}, {"y": 4, "x": 4}])
@pytest.mark.parametrize("sharded", [True, False])
def test_shard_regions_match_the_stores(
    tmp_path: Path, unit: dict[str, int], sharded: bool
) -> None:
    ds = xr.Dataset(
        {"red": (("y", "x"), np.zeros((22, 31), dtype="float32"))},
        coords=xr_coords(GeoBox.from_bbox((0, 0, 310, 220), crs=CRS_31N, resolution=10)),
    ).chunk({"y": 2, "x": 2})
    store = tmp_path / "store.zarr"
    if sharded:
        prepare_template(ds, store, shards=unit, chunks={"y": 2, "x": 2})
    else:
        prepare_template(ds, store, chunks=unit)

    assert shard_regions((22, 31), unit) == write_regions(store, "red")


@pytest.mark.parametrize(
    ("when", "lon", "expected"),
    [
        (
            datetime.datetime(2024, 3, 1, 23, 30, tzinfo=datetime.UTC),
            20.0,
            datetime.date(2024, 3, 2),
        ),
        (
            datetime.datetime(2024, 3, 1, 23, 30, tzinfo=datetime.UTC),
            10.0,
            datetime.date(2024, 3, 1),
        ),
        (
            datetime.datetime(2024, 3, 1, 0, 30, tzinfo=datetime.UTC),
            -20.0,
            datetime.date(2024, 2, 29),
        ),
        (
            datetime.datetime(2024, 3, 1, 0, 30, tzinfo=datetime.UTC),
            -10.0,
            datetime.date(2024, 3, 1),
        ),
    ],
)
def test_solar_day_shifts_by_whole_hours_of_longitude(
    when: datetime.datetime, lon: float, expected: datetime.date
) -> None:
    item = pystac.Item(id="i", geometry=None, bbox=None, datetime=when, properties={})

    assert solar_day(item, lon) == expected


def test_solar_day_falls_back_to_start_datetime() -> None:
    item = pystac.Item(
        id="i",
        geometry=None,
        bbox=None,
        datetime=None,
        properties={},
        start_datetime=datetime.datetime(2024, 3, 1, 23, 30, tzinfo=datetime.UTC),
        end_datetime=datetime.datetime(2024, 3, 5, tzinfo=datetime.UTC),
    )

    assert solar_day(item, 20.0) == datetime.date(2024, 3, 2)


def test_coverage_counts_items_and_solar_days_per_shard() -> None:
    geobox = _geobox()
    # Kept clear of the neighbouring shards, which pick up an item within
    # `_FOOTPRINT_PAD_PX` of them.
    in_first_shard = cast("GeoBox", geobox[0:5, 0:5])
    in_last_shard = cast("GeoBox", geobox[15:20, 25:30])
    items = [
        _item("everywhere_a", geobox, _day(1, 9)),
        _item("everywhere_b", geobox, _day(1, 11)),
        _item("first_shard", in_first_shard, _day(2)),
        _item("last_shard", in_last_shard, _day(3)),
    ]

    gdf = coverage({CRS_31N: (geobox, items)}, {"y": 10, "x": 10}).set_index(["y", "x"])

    assert gdf.crs == "EPSG:4326"
    assert set(gdf["epsg"]) == {32631}
    counts = {key: row.tolist() for key, row in gdf[["n_items", "n_timesteps"]].iterrows()}
    assert counts == {
        (0, 0): [3, 2],
        (0, 10): [2, 1],
        (0, 20): [2, 1],
        (10, 0): [2, 1],
        (10, 10): [2, 1],
        (10, 20): [3, 2],
    }
    assert "n_ascending" not in gdf.columns
    assert not gdf["skipped"].any()
    assert gdf.loc[(0, 0), "geometry"].equals_exact(
        cast("GeoBox", geobox[0:10, 0:10]).extent.to_crs("EPSG:4326").geom, 1e-9
    )


def test_coverage_reports_zero_for_a_shard_no_item_covers() -> None:
    geobox = _geobox()
    items = [_item("first_shard", cast("GeoBox", geobox[0:10, 0:5]), _day(1))]

    gdf = coverage({CRS_31N: (geobox, items)}, {"y": 10, "x": 10}).set_index(["y", "x"])

    assert list(gdf.loc[(0, 0), ["n_items", "n_timesteps"]]) == [1, 1]
    assert list(gdf.loc[(10, 20), ["n_items", "n_timesteps"]]) == [0, 0]


def test_coverage_counts_orbit_states_when_split() -> None:
    geobox = _geobox()
    items = [
        _item("a1", geobox, _day(1), properties={"sat:orbit_state": "ascending"}),
        _item("a2", geobox, _day(2), properties={"sat:orbit_state": "ascending"}),
        _item("d1", geobox, _day(3), properties={"sat:orbit_state": "descending"}),
    ]

    gdf = coverage({CRS_31N: (geobox, items)}, {"y": 10, "x": 10}, split_orbit_states=True)

    assert (gdf["n_ascending"] == 2).all()
    assert (gdf["n_descending"] == 1).all()
    assert (gdf["n_timesteps"] == 3).all()


def test_coverage_flags_skipped_shards() -> None:
    geobox = _geobox()
    items = [_item("everywhere", geobox, _day(1))]

    def skip(crs: CRS, zone: GeoBox):  # noqa: ANN202
        assert (crs, zone) == (CRS_31N, geobox)
        return lambda region: region["x"].start == 20

    gdf = coverage({CRS_31N: (geobox, items)}, {"y": 10, "x": 10}, skip=skip)

    assert gdf.loc[gdf["skipped"], "x"].tolist() == [20, 20]
    assert (gdf["n_items"] == 1).all()
