import datetime

import numpy as np
import pystac
import pytest
import xarray as xr
from odc.geo.crs import CRS
from odc.geo.geobox import GeoBox

from gfetch.mosaic import (
    _pin_mask_band_resampling,
    group_by_utm_zone,
    mosaic,
    mosaic_by_zone,
    zone_geobox,
)
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


def test_group_by_utm_zone_splits_items_by_overlapping_zone() -> None:
    west = _item("west", (29.5, -6.0, 29.6, -5.9))  # zone 35S
    east = _item("east", (39.0, -6.0, 39.1, -5.9))  # zone 37S
    aoi_bbox = (29.0, -6.5, 40.0, -5.5)  # spans zones 35S/36S/37S

    groups = group_by_utm_zone([west, east], aoi_bbox)

    # zone 36S (30-36E) is a candidate the AOI spans, but no item overlaps it
    assert set(groups) == {CRS("EPSG:32735"), CRS("EPSG:32737")}
    assert [it.id for it in groups[CRS("EPSG:32735")]] == ["west"]
    assert [it.id for it in groups[CRS("EPSG:32737")]] == ["east"]


def test_group_by_utm_zone_keeps_same_zone_items_together() -> None:
    a = _item("a", (29.5, -6.0, 29.6, -5.9))
    b = _item("b", (29.6, -6.1, 29.7, -6.0))
    aoi_bbox = (29.0, -6.5, 29.9, -5.5)  # stays inside zone 35S's own band (24-30E)

    groups = group_by_utm_zone([a, b], aoi_bbox)

    assert len(groups) == 1
    assert {it.id for it in next(iter(groups.values()))} == {"a", "b"}


def test_group_by_utm_zone_item_spanning_zones_appears_in_both() -> None:
    # Not possible for Sentinel-2's MGRS-tiled items, routine for e.g. Sentinel-1 GRD,
    # which is delivered in EPSG:4326 rather than tiled to a single native UTM zone.
    wide = _item("wide", (28.0, -6.0, 32.0, -5.0))  # crosses the 35S/36S boundary at 30E
    aoi_bbox = (27.0, -6.5, 33.0, -5.5)

    groups = group_by_utm_zone([wide], aoi_bbox)

    assert set(groups) == {CRS("EPSG:32735"), CRS("EPSG:32736")}
    assert [it.id for it in groups[CRS("EPSG:32735")]] == ["wide"]
    assert [it.id for it in groups[CRS("EPSG:32736")]] == ["wide"]


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


@pytest.mark.parametrize(
    ("resampling", "mask_band", "expected"),
    [
        (None, None, None),
        (None, "scl", {"scl": "nearest"}),
        ("bilinear", "scl", {"*": "bilinear", "scl": "nearest"}),
        ({"nir09": "cubic"}, "scl", {"nir09": "cubic", "scl": "nearest"}),
        # user-requested resampling for the mask band itself is overridden
        ({"scl": "bilinear"}, "scl", {"scl": "nearest"}),
    ],
)
def test_pin_mask_band_resampling(
    resampling: str | dict[str, str] | None,
    mask_band: str | None,
    expected: str | dict[str, str] | None,
) -> None:
    assert _pin_mask_band_resampling(resampling, mask_band) == expected


def test_mosaic_pins_mask_band_to_nearest_even_with_explicit_override(monkeypatch) -> None:
    captured: dict = {}

    def fake_load(items, geobox, bands, *, groupby, chunks, resampling):
        captured["resampling"] = resampling
        return xr.Dataset(
            {
                "red": (("time", "y", "x"), np.array([[[1.0]]])),
                "scl": (("time", "y", "x"), np.array([[[4]]])),
            },
            coords={"time": [datetime.datetime(2020, 6, 6, tzinfo=datetime.UTC)]},
        )

    monkeypatch.setattr("gfetch.mosaic.load", fake_load)

    mosaic(
        [],
        GeoBox.from_bbox((0, 0, 1, 1), crs="EPSG:4326", shape=(1, 1)),
        ["red"],
        mask_band="scl",
        mask_out=frozenset({0}),
        resampling={"*": "bilinear", "scl": "cubic"},
    )

    assert captured["resampling"] == {"*": "bilinear", "scl": "nearest"}


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


@pytest.mark.slow
def test_mosaic_by_zone_sentinel1_against_earthsearch() -> None:
    # Wide enough AOI that real Sentinel-1 GRD scenes (delivered in EPSG:4326, often
    # spanning several degrees of longitude) plausibly straddle a UTM zone boundary -
    # exercises group_by_utm_zone's overlap-based (not one-native-zone-per-item)
    # assignment against real data, not just synthetic bboxes.
    bbox = (2.0, 48.6, 2.6, 49.0)
    source = get_source("earthsearch")
    profile = get_profile("sentinel-1")

    items = search(
        source,
        "sentinel-1",
        bbox=bbox,
        datetime="2026-06-01/2026-06-15",
        query={"sat:orbit_state": {"eq": "ascending"}},
    )
    assert items

    zone_datasets = mosaic_by_zone(
        items,
        bbox,
        list(profile.default_bands)[:1],
        resolution=200.0,  # coarse, keeps it fast
        mask_band=profile.cloud_mask_band,
        mask_out=profile.cloud_mask_out,
    )

    assert zone_datasets
    for crs, ds in zone_datasets.items():
        computed = ds.compute()
        assert "vv" in computed.data_vars
        assert "time" not in computed.dims  # composited away
        assert np.isfinite(computed["vv"].values).any(), f"{crs}: all-NaN composite"
