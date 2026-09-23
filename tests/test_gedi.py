from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import sliderule.sliderule as sliderule_core

from gfetch import gedi as gedi_module
from gfetch.gedi import (
    GEDI_L2A_DEFAULT_FIELDS,
    GEDI_L2A_RH_PERCENTILES,
    _bbox_to_poly,
    expand_rh,
    fetch_gedi_l2a,
    split_bbox,
)


def test_session_init_defaults_trust_env_to_true() -> None:
    """Importing gfetch.gedi patches `Session.__init__` to default `trust_env=True`
    (see the module-level comment in `gfetch/gedi.py`) - a plain, unrelated
    `Session()` construction must pick this up without the caller asking for it,
    and an explicit override must still win.
    """
    from sliderule.session import Session

    assert Session(cluster=None).session.trust_env is True
    assert Session(cluster=None, trust_env=False).session.trust_env is False


def test_bbox_to_poly_is_closed_ccw_ring() -> None:
    poly = _bbox_to_poly((2.0, 48.0, 3.0, 49.0))

    assert poly[0] == poly[-1]
    assert [p["lon"] for p in poly] == [2.0, 3.0, 3.0, 2.0, 2.0]
    assert [p["lat"] for p in poly] == [48.0, 48.0, 49.0, 49.0, 48.0]


def test_split_bbox_small_bbox_is_single_tile() -> None:
    bbox = (2.0, 48.0, 2.05, 48.05)

    assert split_bbox(bbox) == [bbox]


@pytest.mark.parametrize("bbox", [(2.0, 48.0, 3.0, 49.0), (30.0, -12.0, 32.0, 2.0)])
def test_split_bbox_tiles_cover_bbox_within_max_size(
    bbox: tuple[float, float, float, float],
) -> None:
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    tiles = split_bbox(bbox, max_size_m=10_000)

    assert min(t[0] for t in tiles) == bbox[0]
    assert min(t[1] for t in tiles) == bbox[1]
    assert max(t[2] for t in tiles) == bbox[2]
    assert max(t[3] for t in tiles) == bbox[3]
    total_area = sum((t[2] - t[0]) * (t[3] - t[1]) for t in tiles)
    assert total_area == pytest.approx((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    for min_lon, min_lat, max_lon, max_lat in tiles:
        widest_lat = 0.0 if min_lat <= 0.0 <= max_lat else min(min_lat, max_lat, key=abs)
        assert geod.line_length([min_lon, max_lon], [widest_lat, widest_lat]) <= 10_000
        assert geod.line_length([min_lon, min_lon], [min_lat, max_lat]) <= 10_000


def test_fetch_gedi_l2a_splits_into_tiles_without_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Footprints on an edge shared by two tiles are returned by both SlideRule
    requests, but must appear only once in the result.
    """
    bbox = (2.0, 48.0, 2.2, 48.2)
    lons = [2.0, 2.1, 2.2, 2.05]
    lats = [48.0, 48.1, 48.2, 48.15]
    requested_polys = []

    def fake_gedi02ap(parms: dict) -> gpd.GeoDataFrame:
        poly = parms["poly"]
        requested_polys.append(poly)
        min_lon, max_lon = poly[0]["lon"], poly[2]["lon"]
        min_lat, max_lat = poly[0]["lat"], poly[2]["lat"]
        inside = [
            (lon, lat)
            for lon, lat in zip(lons, lats, strict=True)
            if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat
        ]
        if not inside:
            return sliderule_core.emptyframe(crs="EPSG:7912")
        xs, ys = zip(*inside, strict=True)
        return gpd.GeoDataFrame(
            {"elevation_lm": list(xs), "elevation_hr": list(ys)},
            geometry=gpd.points_from_xy(xs, ys),
            crs="EPSG:7912",
        )

    monkeypatch.setattr(gedi_module.sliderule, "init", lambda *args, **kwargs: None)
    monkeypatch.setattr(gedi_module.gedi, "gedi02ap", fake_gedi02ap)

    gdf = fetch_gedi_l2a(bbox, max_size_m=10_000)

    assert len(requested_polys) == len(split_bbox(bbox, 10_000)) > 1
    assert sorted(zip(gdf.geometry.x, gdf.geometry.y, strict=True)) == sorted(
        zip(lons, lats, strict=True)
    )
    assert gdf["elevation_lm"].dtype == float


def _fake_gedi02ap_over(
    points: list[tuple[float, float]], calls: list[dict]
) -> Callable[[dict], gpd.GeoDataFrame]:
    """A `gedi02ap` stand-in returning whichever of `points` fall in the request's
    polygon, as an empty frame (like SlideRule's) when none do.
    """

    def fake_gedi02ap(parms: dict) -> gpd.GeoDataFrame:
        calls.append(parms)
        poly = parms["poly"]
        min_lon, max_lon = poly[0]["lon"], poly[2]["lon"]
        min_lat, max_lat = poly[0]["lat"], poly[2]["lat"]
        inside = [
            (lon, lat)
            for lon, lat in points
            if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat
        ]
        if not inside:
            return sliderule_core.emptyframe(crs="EPSG:7912")
        xs, ys = zip(*inside, strict=True)
        return gpd.GeoDataFrame(
            {"elevation_lm": list(xs), "elevation_hr": list(ys)},
            geometry=gpd.points_from_xy(xs, ys),
            crs="EPSG:7912",
        )

    return fake_gedi02ap


def test_fetch_gedi_l2a_resumes_from_saved_tiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bbox = (2.0, 48.0, 2.2, 48.2)
    n_tiles = len(split_bbox(bbox, 10_000))
    points = [(2.01, 48.01), (2.19, 48.19)]
    tile_dir = tmp_path / "l2a.parquet.tiles"
    monkeypatch.setattr(gedi_module.sliderule, "init", lambda *args, **kwargs: None)

    calls: list[dict] = []
    fake = _fake_gedi02ap_over(points, calls)

    def interrupted(parms: dict) -> gpd.GeoDataFrame:
        if len(calls) == 2:
            raise KeyboardInterrupt
        return fake(parms)

    monkeypatch.setattr(gedi_module.gedi, "gedi02ap", interrupted)
    with pytest.raises(KeyboardInterrupt):
        fetch_gedi_l2a(bbox, max_size_m=10_000, tile_dir=tile_dir)
    assert len(list(tile_dir.glob("[!.]*.parquet"))) == 2

    calls.clear()
    monkeypatch.setattr(gedi_module.gedi, "gedi02ap", fake)
    resumed = fetch_gedi_l2a(bbox, max_size_m=10_000, tile_dir=tile_dir)

    assert len(calls) == n_tiles - 2
    expected = fetch_gedi_l2a(bbox, max_size_m=10_000)
    assert sorted(zip(resumed.geometry.x, resumed.geometry.y, strict=True)) == sorted(points)
    assert list(resumed.columns) == list(expected.columns)
    assert resumed["elevation_lm"].dtype == expected["elevation_lm"].dtype


def test_fetch_gedi_l2a_refuses_tiles_from_other_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bbox = (2.0, 48.0, 2.05, 48.05)
    tile_dir = tmp_path / "tiles"
    monkeypatch.setattr(gedi_module.sliderule, "init", lambda *args, **kwargs: None)
    monkeypatch.setattr(gedi_module.gedi, "gedi02ap", _fake_gedi02ap_over([], []))
    fetch_gedi_l2a(
        bbox, time_range=("2020-01-01T00:00:00Z", "2020-12-31T23:59:59Z"), tile_dir=tile_dir
    )

    with pytest.raises(ValueError, match="delete it"):
        fetch_gedi_l2a(
            bbox, time_range=("2021-01-01T00:00:00Z", "2021-12-31T23:59:59Z"), tile_dir=tile_dir
        )


def test_fetch_gedi_l2a_handles_empty_or_failed_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """`sliderule.gedi.gedi02ap()` returns a bare-geometry frame with no data
    columns both on a genuine zero-footprint match and on a server-side request
    failure (e.g. "Unexpected termination of response", confirmed live) - the
    column selection below must not crash on either.
    """
    monkeypatch.setattr(gedi_module.sliderule, "init", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gedi_module.gedi, "gedi02ap", lambda parms: sliderule_core.emptyframe(crs="EPSG:7912")
    )

    gdf = fetch_gedi_l2a((2.0, 48.0, 2.1, 48.1))

    assert len(gdf) == 0
    assert list(gdf.columns) == [*GEDI_L2A_DEFAULT_FIELDS, gdf.geometry.name]


def test_fetch_gedi_l2a_missing_field_on_nonempty_response_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuinely wrong field name against a non-empty response should still
    raise clearly, unlike the empty-response case above.
    """
    monkeypatch.setattr(gedi_module.sliderule, "init", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gedi_module.gedi,
        "gedi02ap",
        lambda parms: gpd.GeoDataFrame(
            {"elevation_lm": [1.0]}, geometry=gpd.points_from_xy([2.0], [48.0]), crs="EPSG:7912"
        ),
    )

    with pytest.raises(KeyError, match="typo_field"):
        fetch_gedi_l2a((2.0, 48.0, 2.1, 48.1), fields=("elevation_lm", "typo_field"))


def test_fetch_gedi_l2a_passes_anc_fields_to_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_parms = {}

    def fake_gedi02ap(parms: dict) -> gpd.GeoDataFrame:
        captured_parms.update(parms)
        return gpd.GeoDataFrame(
            {"elevation_lm": [1.0], "elevation_hr": [1.0], "quality_flag": [1]},
            geometry=gpd.points_from_xy([2.0], [48.0]),
            crs="EPSG:7912",
        )

    monkeypatch.setattr(gedi_module.sliderule, "init", lambda *args, **kwargs: None)
    monkeypatch.setattr(gedi_module.gedi, "gedi02ap", fake_gedi02ap)

    fetch_gedi_l2a((2.0, 48.0, 2.1, 48.1), anc_fields=["quality_flag"])

    assert captured_parms["anc_fields"] == ["quality_flag"]


def test_fetch_gedi_l2a_keeps_anc_fields_regardless_of_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`anc_fields` columns aren't part of `gedi02ap`'s fixed schema, so the
    `fields` filter must not drop them.
    """
    monkeypatch.setattr(gedi_module.sliderule, "init", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gedi_module.gedi,
        "gedi02ap",
        lambda parms: gpd.GeoDataFrame(
            {"elevation_lm": [1.0], "quality_flag": [1]},
            geometry=gpd.points_from_xy([2.0], [48.0]),
            crs="EPSG:7912",
        ),
    )

    gdf = fetch_gedi_l2a(
        (2.0, 48.0, 2.1, 48.1), fields=("elevation_lm",), anc_fields=["quality_flag"]
    )

    assert list(gdf.columns) == ["elevation_lm", "quality_flag", gdf.geometry.name]


def test_expand_rh_splits_percentile_array_into_named_columns() -> None:
    rh_array = np.arange(101, dtype=float)
    gdf = gpd.GeoDataFrame(
        {"rh": [rh_array]}, geometry=gpd.points_from_xy([2.0], [48.0]), crs="EPSG:7912"
    )

    expanded = expand_rh(gdf, percentiles=(0, 50, 100))

    assert "rh" not in expanded.columns
    assert expanded["rh0"].tolist() == [0.0]
    assert expanded["rh50"].tolist() == [50.0]
    assert expanded["rh100"].tolist() == [100.0]


@pytest.mark.slow
def test_fetch_gedi_l2a_returns_footprints_over_france() -> None:
    bbox = (2.55, 48.35, 2.75, 48.50)

    gdf = fetch_gedi_l2a(bbox, time_range=("2020-01-01T00:00:00Z", "2020-12-31T23:59:59Z"))

    assert len(gdf) > 0
    assert list(gdf.columns) == [*GEDI_L2A_DEFAULT_FIELDS, gdf.geometry.name]
    minx, miny, maxx, maxy = gdf.total_bounds
    assert bbox[0] <= minx
    assert maxx <= bbox[2]
    assert bbox[1] <= miny
    assert maxy <= bbox[3]


@pytest.mark.slow
def test_fetch_gedi_l2a_fields_none_keeps_every_column() -> None:
    gdf = fetch_gedi_l2a(
        (2.55, 48.35, 2.75, 48.50),
        time_range=("2020-01-01T00:00:00Z", "2020-12-31T23:59:59Z"),
        fields=None,
    )

    for field in ("orbit", "solar_elevation", "track", "sensitivity", "flags", "beam"):
        assert field in gdf.columns


@pytest.mark.slow
def test_fetch_gedi_l2a_anc_fields_and_expand_rh_over_france() -> None:
    anc_fields = [
        "rh",
        "quality_flag",
        "degrade_flag",
        "surface_flag",
        "digital_elevation_model",
        "elevation_bias_flag",
        "energy_total",
        "num_detectedmodes",
        "selected_algorithm",
        "selected_mode",
        "selected_mode_flag",
        "delta_time",
        "solar_azimuth",
        "land_cover_data/landsat_treecover",
        "land_cover_data/modis_treecover",
    ]

    gdf = fetch_gedi_l2a(
        (2.55, 48.35, 2.75, 48.50),
        time_range=("2020-06-01T00:00:00Z", "2020-06-15T23:59:59Z"),
        anc_fields=anc_fields,
    )
    gdf = expand_rh(gdf)

    assert len(gdf) > 0
    for field in [*anc_fields[1:], *(f"rh{p}" for p in GEDI_L2A_RH_PERCENTILES)]:
        assert field in gdf.columns
    assert "rh" not in gdf.columns
    assert gdf["quality_flag"].isin([0, 1]).all()
