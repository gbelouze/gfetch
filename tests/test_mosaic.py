import datetime

import numpy as np
import pystac
import pytest
from odc.geo.crs import CRS
from odc.geo.geobox import GeoBox

from gfetch.mosaic import group_by_utm_zone, mosaic, mosaic_by_zone, zone_geobox
from gfetch.profiles import get_profile
from gfetch.search import search
from gfetch.sources import get_source


def _item(item_id: str, bbox: tuple[float, float, float, float]) -> pystac.Item:
    return pystac.Item(
        id=item_id,
        geometry=None,
        bbox=list(bbox),
        datetime=datetime.datetime(2020, 6, 6, tzinfo=datetime.UTC),
        properties={},
    )


def test_group_by_utm_zone_splits_items_by_native_zone() -> None:
    west = _item("west", (29.5, -6.0, 29.6, -5.9))  # zone 35S
    east = _item("east", (39.0, -6.0, 39.1, -5.9))  # zone 37S

    groups = group_by_utm_zone([west, east])

    assert set(groups) == {CRS("EPSG:32735"), CRS("EPSG:32737")}
    assert [it.id for it in groups[CRS("EPSG:32735")]] == ["west"]
    assert [it.id for it in groups[CRS("EPSG:32737")]] == ["east"]


def test_group_by_utm_zone_keeps_same_zone_items_together() -> None:
    a = _item("a", (29.5, -6.0, 29.6, -5.9))
    b = _item("b", (29.6, -6.1, 29.7, -6.0))

    groups = group_by_utm_zone([a, b])

    assert len(groups) == 1
    assert {it.id for it in next(iter(groups.values()))} == {"a", "b"}


def test_zone_geobox_clips_aoi_to_zone_band() -> None:
    # AOI spans two zones (boundary at 30 degrees E); zone 35S covers 24-30E.
    aoi_bbox = (29.0, -6.0, 31.0, -5.0)

    geobox = zone_geobox(CRS("EPSG:32735"), aoi_bbox, resolution=1000.0)
    clipped = geobox.boundingbox.to_crs("EPSG:4326")

    # A few km of slack: coarse (1km) pixel snapping plus reprojection distortion at
    # the box's edges, not an exact round-trip.
    assert clipped.left == pytest.approx(29.0, abs=0.05)
    assert clipped.right == pytest.approx(30.0, abs=0.05)
    assert clipped.right < 31.0  # stayed clipped to the 35S zone, not the full AOI


@pytest.mark.slow
def test_mosaic_against_earthsearch() -> None:
    bbox = (2.30, 48.85, 2.33, 48.87)  # small AOI, keeps the test fast
    source = get_source("earthsearch")
    profile = get_profile("sentinel-2")

    items = search(
        source,
        "sentinel-2",
        bbox=bbox,
        datetime="2026-06-01/2026-06-30",
        query={"eo:cloud_cover": {"lt": 40}},
    )
    assert items

    geobox = GeoBox.from_bbox(bbox, crs="utm", resolution=60.0)  # coarse, keeps it fast
    ds = mosaic(
        items,
        geobox,
        ["red"],
        mask_band=profile.cloud_mask_band,
        mask_out=profile.cloud_mask_out,
    )
    computed = ds.compute()

    assert "red" in computed.data_vars
    assert "time" not in computed.dims  # composited away
    assert np.isfinite(computed["red"].values).any()


@pytest.mark.slow
def test_mosaic_by_zone_against_earthsearch() -> None:
    bbox = (2.30, 48.85, 2.33, 48.87)  # small, single-zone AOI, keeps the test fast
    source = get_source("earthsearch")
    profile = get_profile("sentinel-2")

    items = search(
        source,
        "sentinel-2",
        bbox=bbox,
        datetime="2026-06-01/2026-06-30",
        query={"eo:cloud_cover": {"lt": 40}},
    )
    assert items

    zone_datasets = mosaic_by_zone(
        items,
        bbox,
        ["red"],
        resolution=60.0,  # coarse, keeps it fast
        mask_band=profile.cloud_mask_band,
        mask_out=profile.cloud_mask_out,
    )

    assert list(zone_datasets) == [CRS("EPSG:32631")]  # Paris is in UTM zone 31N
    computed = next(iter(zone_datasets.values())).compute()
    assert "red" in computed.data_vars
    assert np.isfinite(computed["red"].values).any()
