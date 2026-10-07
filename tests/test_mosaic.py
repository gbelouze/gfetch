import datetime
from pathlib import Path

import numpy as np
import pystac
import pytest
import rasterio
import shapely
import xarray as xr
from odc.geo.crs import CRS
from odc.geo.geobox import GeoBox
from odc.geo.geom import box
from rasterio.transform import from_origin

from gfetch.mosaic import (
    _items_intersecting,
    _pin_mask_band_resampling,
    group_by_utm_zone,
    mask_nodata,
    mosaic,
    orbit_state_variables,
    outside_aoi,
    resolve_chunks,
    resolve_compute_chunks,
    resolve_shards,
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
    aoi_bbox = (28.0, -6.5, 40.0, -5.5)  # spans zones 35S/36S/37S

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
    aoi_bbox = (28.0, -6.0, 32.0, -5.0)

    geobox = zone_geobox(CRS("EPSG:32735"), aoi_bbox, resolution=1000.0)
    clipped = geobox.boundingbox.to_crs("EPSG:4326")

    # A few km of slack: coarse (1km) pixel snapping plus reprojection distortion at
    # the box's edges, not an exact round-trip.
    assert clipped.left == pytest.approx(28.0, abs=0.05)
    assert clipped.right == pytest.approx(30.0, abs=0.05)
    assert clipped.right < 32.0  # stayed clipped to the 35S zone, not the full AOI


def test_thin_end_zone_strip_goes_to_its_neighbour_zone() -> None:
    west = _item("west", (29.5, -6.0, 29.6, -5.9))  # zone 35S, 0.66 degree strip
    east = _item("east", (39.0, -6.0, 39.1, -5.9))  # zone 37S
    aoi_bbox = (29.34, -9.0, 40.85, -1.0)  # spans zones 35S/36S/37S

    groups = group_by_utm_zone([west, east], aoi_bbox)

    assert set(groups) == {CRS("EPSG:32736"), CRS("EPSG:32737")}
    assert [it.id for it in groups[CRS("EPSG:32736")]] == ["west"]
    geobox = zone_geobox(CRS("EPSG:32736"), aoi_bbox, resolution=1000.0)
    extended = geobox.boundingbox.to_crs("EPSG:4326")
    assert extended.left == pytest.approx(29.34, abs=0.05)
    assert extended.right == pytest.approx(36.0, abs=0.05)


def test_two_thin_zones_merge_into_the_wider_one() -> None:
    aoi_bbox = (29.6, -6.0, 30.3, -5.0)  # 0.4 degree in 35S, 0.3 degree in 36S
    item = _item("a", aoi_bbox)

    groups = group_by_utm_zone([item], aoi_bbox)

    assert set(groups) == {CRS("EPSG:32735")}
    geobox = zone_geobox(CRS("EPSG:32735"), aoi_bbox, resolution=100.0)
    merged = geobox.boundingbox.to_crs("EPSG:4326")
    assert merged.left == pytest.approx(29.6, abs=0.01)
    assert merged.right == pytest.approx(30.3, abs=0.01)


def test_thin_zone_strip_merges_in_both_hemispheres() -> None:
    aoi_bbox = (29.5, -1.0, 33.0, 1.0)  # crosses the equator and the 35/36 boundary
    item = _item("a", aoi_bbox)

    groups = group_by_utm_zone([item], aoi_bbox)

    assert set(groups) == {CRS("EPSG:32636"), CRS("EPSG:32736")}


def test_single_thin_zone_is_kept() -> None:
    aoi_bbox = (29.5, -6.0, 29.7, -5.0)
    item = _item("a", aoi_bbox)

    assert set(group_by_utm_zone([item], aoi_bbox)) == {CRS("EPSG:32735")}


def test_outside_aoi_tests_regions_against_the_aoi_polygon() -> None:
    # 2x2 regions of ~0.5 degree each; the triangle covers the south-west one only.
    geobox = GeoBox.from_bbox((29.0, -6.0, 30.0, -5.0), crs="EPSG:4326", resolution=0.01)
    geobox = geobox.to_crs("EPSG:32735")
    half_y, half_x = geobox.shape.y // 2, geobox.shape.x // 2
    aoi = shapely.Polygon([(29.05, -5.95), (29.3, -5.95), (29.05, -5.7)])

    outside = outside_aoi(geobox, aoi)

    top, bottom = slice(0, half_y), slice(half_y, None)
    left, right = slice(0, half_x), slice(half_x, None)
    assert not outside({"y": bottom, "x": left})
    assert outside({"y": bottom, "x": right})
    assert outside({"y": top, "x": left})
    assert outside({"y": top, "x": right})


def test_outside_aoi_disjoint_from_geobox_skips_everything() -> None:
    geobox = GeoBox.from_bbox((29.0, -6.0, 30.0, -5.0), crs="EPSG:4326", resolution=0.01)

    outside = outside_aoi(geobox, shapely.box(35.0, -6.0, 36.0, -5.0))

    assert outside({"y": slice(None), "x": slice(None)})


def test_resolve_chunks_defaults() -> None:
    assert resolve_chunks(None) == {"x": 256, "y": 256, "time": -1}


def test_resolve_chunks_keeps_user_overrides_and_adds_time() -> None:
    assert resolve_chunks({"x": 512, "y": 256}) == {"x": 512, "y": 256, "time": -1}


def test_resolve_chunks_does_not_override_explicit_time() -> None:
    assert resolve_chunks({"x": 512, "y": 256, "time": 3}) == {"x": 512, "y": 256, "time": 3}


def test_resolve_compute_chunks_scales_spatial_chunks() -> None:
    chunks = {"x": 256, "y": 256, "time": -1}

    assert resolve_compute_chunks(chunks, 1, {"x": 8192, "y": 8192}) == chunks
    assert resolve_compute_chunks(chunks, 4, {"x": 8192, "y": 8192}) == {
        "x": 1024,
        "y": 1024,
        "time": -1,
    }


def test_resolve_compute_chunks_defaults_to_4_chunks_per_side() -> None:
    assert resolve_compute_chunks(
        {"x": 256, "y": 256, "time": -1}, None, {"x": 4096, "y": 4096}
    ) == {
        "x": 1024,
        "y": 1024,
        "time": -1,
    }


@pytest.mark.parametrize(("factor", "shards"), [(3, {"x": 8192, "y": 8192}), (2, None)])
def test_resolve_compute_chunks_rejects_bricks_not_dividing_the_write_unit(
    factor: int, shards: dict[str, int] | None
) -> None:
    with pytest.raises(ValueError, match="compute_chunk_factor"):
        resolve_compute_chunks({"x": 256, "y": 256, "time": -1}, factor, shards)


def test_resolve_shards_defaults_to_16_chunks_per_side() -> None:
    assert resolve_shards(None, {"x": 256, "y": 128, "time": -1}) == {"y": 2048, "x": 4096}


def test_resolve_shards_scales_chunks_by_factor() -> None:
    assert resolve_shards(4, {"x": 256, "y": 256}) == {"y": 1024, "x": 1024}


def test_resolve_shards_factor_one_disables_sharding() -> None:
    assert resolve_shards(1, {"x": 256, "y": 256, "time": -1}) is None


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

    def fake_load(items, geobox, bands, *, groupby, chunks, resampling, log_footprint, patch_url):
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


def test_mosaic_overrides_non_default_time_chunk_and_warns(monkeypatch, caplog) -> None:
    """`mosaic()` always reduces over the whole time axis - a non-full `time` chunk
    (e.g. a leftover config value from before `_DEFAULT_TIME_CHUNK` existed) forces
    dask to rechunk internally before reducing, which was confirmed live 2026-09-22
    to also shrink the spatial x/y chunk size as a side effect for a large enough
    zone. `mosaic()` must override it rather than pass it through to `load()`.
    """
    captured: dict = {}

    def fake_load(items, geobox, bands, *, groupby, chunks, resampling, log_footprint, patch_url):
        captured["chunks"] = chunks
        return xr.Dataset(
            {"red": (("time", "y", "x"), np.array([[[1.0]]]))},
            coords={"time": [datetime.datetime(2020, 6, 6, tzinfo=datetime.UTC)]},
        )

    monkeypatch.setattr("gfetch.mosaic.load", fake_load)

    with caplog.at_level("WARNING"):
        mosaic(
            [],
            GeoBox.from_bbox((0, 0, 1, 1), crs="EPSG:4326", shape=(1, 1)),
            ["red"],
            chunks={"x": 512, "y": 512, "time": 39},
        )

    assert captured["chunks"] == {"x": 512, "y": 512, "time": -1}
    assert "chunks['time']" in caplog.text


def test_mosaic_pins_output_chunks_to_requested_grid(monkeypatch) -> None:
    """Belt-and-suspenders: even with the time-chunk override above removing the
    known trigger, nothing guarantees `composite()`'s reduction (or any per-band
    resampling upstream) preserves the requested x/y chunk grid for every variable.
    `mosaic()` must pin its output back onto the requested grid regardless of what
    `load()` happened to return.
    """

    def fake_load(items, geobox, bands, *, groupby, chunks, resampling, log_footprint, patch_url):
        time_coord = [datetime.datetime(2020, 6, d, tzinfo=datetime.UTC) for d in (1, 2)]
        red = xr.DataArray(
            np.zeros((2, 4, 4)), dims=("time", "y", "x"), coords={"time": time_coord}
        ).chunk({"time": -1, "y": 3, "x": 3})  # deliberately misaligned vs. requested below
        return xr.Dataset({"red": red})

    monkeypatch.setattr("gfetch.mosaic.load", fake_load)

    ds = mosaic(
        [],
        GeoBox.from_bbox((0, 0, 1, 1), crs="EPSG:4326", shape=(4, 4)),
        ["red"],
        chunks={"x": 2, "y": 2},
    )

    y_chunks, x_chunks = ds["red"].data.chunks
    assert y_chunks == (2, 2)
    assert x_chunks == (2, 2)


def test_mask_nodata_sets_nodata_to_nan_except_excluded() -> None:
    ds = xr.Dataset(
        {
            "vv": ("x", np.array([0, 5], dtype=np.uint16), {"nodata": 0, "units": "dB"}),
            "scl": ("x", np.array([0, 4], dtype=np.uint8), {"nodata": 0}),
            "no_attr": ("x", np.array([0, 1])),
        }
    )

    masked = mask_nodata(ds, exclude="scl")

    np.testing.assert_array_equal(masked["vv"].values, [np.nan, 5.0])
    assert masked["vv"].attrs == {"units": "dB"}
    assert masked["scl"].identical(ds["scl"])
    assert masked["no_attr"].identical(ds["no_attr"])


def test_mosaic_composite_ignores_nodata(monkeypatch) -> None:
    """A median over nodata-filled time steps must only see the valid ones, and a
    pixel no time step covers must come out NaN, not `nodata`."""

    def fake_load(items, geobox, bands, *, groupby, chunks, resampling, log_footprint, patch_url):
        time_coord = [datetime.datetime(2020, 6, d, tzinfo=datetime.UTC) for d in (1, 2, 3)]
        vv = np.array([[[0, 0]], [[0, 10]], [[0, 30]]], dtype=np.uint16)
        return xr.Dataset(
            {"vv": (("time", "y", "x"), vv, {"nodata": 0})}, coords={"time": time_coord}
        )

    monkeypatch.setattr("gfetch.mosaic.load", fake_load)

    ds = mosaic([], GeoBox.from_bbox((0, 0, 1, 1), crs="EPSG:4326", shape=(1, 2)), ["vv"])

    np.testing.assert_array_equal(ds["vv"].values, [[np.nan, 20.0]])


def _orbit_item(item_id: str, orbit_state: str | None) -> pystac.Item:
    item = _item(item_id, (0, 0, 1, 1))
    if orbit_state is not None:
        item.properties["sat:orbit_state"] = orbit_state
    return item


def _fake_load_by_item_count(monkeypatch) -> list[list[str]]:
    """Patch `load` to fill every band with the number of items it was given."""
    calls: list[list[str]] = []

    def fake_load(items, geobox, bands, *, groupby, chunks, resampling, log_footprint, patch_url):
        calls.append([item.id for item in items])
        value = np.full((1, 2, 2), float(len(items)))
        return xr.Dataset(
            {band: (("time", "y", "x"), value) for band in bands},
            coords={"time": [datetime.datetime(2020, 6, 6, tzinfo=datetime.UTC)]},
        )

    monkeypatch.setattr("gfetch.mosaic.load", fake_load)
    return calls


def test_mosaic_split_orbit_states_composites_each_separately(monkeypatch) -> None:
    calls = _fake_load_by_item_count(monkeypatch)
    items = [
        _orbit_item("a1", "ascending"),
        _orbit_item("d1", "descending"),
        _orbit_item("a2", "ascending"),
    ]

    ds = mosaic(
        items,
        GeoBox.from_bbox((0, 0, 1, 1), crs="EPSG:4326", shape=(2, 2)),
        ["vv", "vh"],
        split_orbit_states=True,
    )

    assert calls == [["a1", "a2"], ["d1"]]
    assert list(ds.data_vars) == orbit_state_variables(["vv", "vh"])
    assert (ds["vh_ascending"].values == 2).all()
    assert (ds["vh_descending"].values == 1).all()


def test_mosaic_split_orbit_states_fills_missing_state_with_nan(monkeypatch) -> None:
    calls = _fake_load_by_item_count(monkeypatch)

    ds = mosaic(
        [_orbit_item("d1", "descending")],
        GeoBox.from_bbox((0, 0, 1, 1), crs="EPSG:4326", shape=(2, 2)),
        ["vv"],
        split_orbit_states=True,
    )

    assert calls == [["d1"]]
    assert list(ds.data_vars) == ["vv_ascending", "vv_descending"]
    assert np.isnan(ds["vv_ascending"].values).all()
    assert (ds["vv_descending"].values == 1).all()


@pytest.mark.parametrize("orbit_state", [None, "sideways"])
def test_mosaic_split_orbit_states_rejects_unknown_orbit_state(
    monkeypatch, orbit_state: str | None
) -> None:
    _fake_load_by_item_count(monkeypatch)

    with pytest.raises(ValueError, match="sat:orbit_state"):
        mosaic(
            [_orbit_item("x", orbit_state)],
            GeoBox.from_bbox((0, 0, 1, 1), crs="EPSG:4326", shape=(2, 2)),
            ["vv"],
            split_orbit_states=True,
        )


_UTM_X0, _UTM_Y0, _TILE_RES, _TILE_PX = 500_000, 5_400_000, 100, 100


def _tile_item(tmp_path: Path, item_id: str, col: int, day: int, seed: int) -> pystac.Item:
    """An item backed by a local GeoTIFF, the `col`-th 10 km tile east of a fixed
    EPSG:32631 origin."""
    left = _UTM_X0 + col * _TILE_RES * _TILE_PX
    top = _UTM_Y0 + _TILE_RES * _TILE_PX
    path = tmp_path / f"{item_id}.tif"
    data = np.random.default_rng(seed).integers(1, 1000, (_TILE_PX, _TILE_PX), dtype=np.uint16)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=_TILE_PX,
        height=_TILE_PX,
        count=1,
        dtype="uint16",
        crs="EPSG:32631",
        transform=from_origin(left, top, _TILE_RES, _TILE_RES),
        nodata=0,
    ) as dst:
        dst.write(data, 1)
    footprint = box(left, _UTM_Y0, left + _TILE_RES * _TILE_PX, top, "EPSG:32631")
    footprint = footprint.to_crs("EPSG:4326")
    item = pystac.Item(
        id=item_id,
        geometry=footprint.json,
        bbox=list(footprint.boundingbox),
        datetime=datetime.datetime(2020, 6, day, tzinfo=datetime.UTC),
        properties={},
    )
    item.add_asset("red", pystac.Asset(str(path), media_type=pystac.MediaType.GEOTIFF))
    return item


def _geobox_in_first_tile() -> GeoBox:
    # Right edge 50 m short of the second tile, which must be kept but not the sixth.
    bbox = (_UTM_X0 + 3000, _UTM_Y0 + 2000, _UTM_X0 + 9950, _UTM_Y0 + 8000)
    return GeoBox.from_bbox(bbox, crs="EPSG:32631", resolution=70)


def test_items_intersecting_drops_disjoint_items(tmp_path: Path) -> None:
    items = [_tile_item(tmp_path, f"c{col}", col, 1, col) for col in (0, 1, 5)]
    no_geometry = _item("no_geometry", (0, 0, 1, 1))

    kept = _items_intersecting([*items, no_geometry], _geobox_in_first_tile())

    assert [item.id for item in kept] == ["c0", "c1", "no_geometry"]


def test_items_intersecting_keeps_first_item_when_none_intersect(tmp_path: Path) -> None:
    items = [_tile_item(tmp_path, f"c{col}", col, 1, col) for col in (5, 6)]

    assert [item.id for item in _items_intersecting(items, _geobox_in_first_tile())] == ["c5"]


@pytest.mark.parametrize(
    ("cols", "n_kept"),
    [
        ((0, 0, 1, 1, 5), 4),
        # no item intersects: a single all-nodata time step, all-NaN composite
        ((5, 6), 1),
    ],
)
def test_mosaic_item_filtering_leaves_composite_unchanged(
    tmp_path: Path, monkeypatch, cols: tuple[int, ...], n_kept: int
) -> None:
    items = [_tile_item(tmp_path, f"i{i}", col, i + 1, i) for i, col in enumerate(cols)]
    geobox = _geobox_in_first_tile()

    def run() -> tuple[xr.Dataset, list[int]]:
        n_times: list[int] = []
        ds = mosaic(
            items,
            geobox,
            ["red"],
            resampling="bilinear",
            on_load=lambda loaded: n_times.append(loaded.sizes["time"]),
        )
        return ds.compute(), n_times

    filtered, filtered_times = run()
    monkeypatch.setattr("gfetch.mosaic._items_intersecting", lambda items, geobox: list(items))
    unfiltered, unfiltered_times = run()

    xr.testing.assert_identical(filtered, unfiltered)
    assert filtered_times == [n_kept]
    assert unfiltered_times == [len(cols)]


def test_mosaic_reads_assets_through_patch_url(tmp_path: Path) -> None:
    item = _tile_item(tmp_path, "i0", 0, 1, 0)
    real_href = item.assets["red"].href
    item.assets["red"].href = "https://unsigned.example/i0.tif"
    patched: list[str] = []

    def patch_url(href: str) -> str:
        patched.append(href)
        return real_href

    ds = mosaic([item], _geobox_in_first_tile(), ["red"], patch_url=patch_url).compute()

    assert patched == ["https://unsigned.example/i0.tif"]
    assert not ds["red"].isnull().all()


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
def test_mosaic_per_zone_against_earthsearch() -> None:
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

    # Composed the same way `cli/mosaic.py` does: group by zone, then mosaic each
    # zone's own geobox independently.
    zones = group_by_utm_zone(items, bbox)
    assert list(zones) == [CRS("EPSG:32631")]  # Paris is in UTM zone 31N

    crs, zone_items = next(iter(zones.items()))
    ds = mosaic(
        zone_items,
        zone_geobox(crs, bbox, resolution=60.0),  # coarse, keeps it fast
        ["red"],
        mask_band=profile.cloud_mask_band,
        mask_out=profile.cloud_mask_out,
    )
    computed = ds.compute()
    assert "red" in computed.data_vars
    assert np.isfinite(computed["red"].values).any()


@pytest.mark.slow
def test_mosaic_per_zone_sentinel1_against_earthsearch() -> None:
    # Wide enough AOI that real Sentinel-1 GRD scenes (delivered in EPSG:4326, often
    # spanning several degrees of longitude) plausibly straddle a UTM zone boundary -
    # exercises group_by_utm_zone's overlap-based (not one-native-zone-per-item)
    # assignment against real data.
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

    zones = group_by_utm_zone(items, bbox)
    assert zones

    for crs, zone_items in zones.items():
        ds = mosaic(
            zone_items,
            zone_geobox(crs, bbox, resolution=200.0),  # coarse, keeps it fast
            list(profile.default_bands)[:1],
            mask_band=profile.cloud_mask_band,
            mask_out=profile.cloud_mask_out,
        )
        computed = ds.compute()
        assert "vv" in computed.data_vars
        assert "time" not in computed.dims  # composited away
        assert np.isfinite(computed["vv"].values).any(), f"{crs}: all-NaN composite"
