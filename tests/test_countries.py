import geopandas as gpd
import pytest
import shapely

from gfetch import countries as countries_module
from gfetch.countries import _suggest, resolve_country_polygon


def _fake_borders() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "name": ["Mozambique", "United Republic of Tanzania", "Mauritania"],
            "geometry": [
                shapely.box(30.0, -27.0, 41.0, -10.0),
                shapely.box(29.0, -12.0, 41.0, -1.0),
                shapely.box(-17.0, 15.0, -5.0, 27.0),
            ],
        },
        crs="EPSG:4326",
    )


def test_resolve_country_polygon_single_country(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(countries_module, "_country_borders", _fake_borders)

    polygon = resolve_country_polygon(["Mozambique"])

    assert polygon.bounds == (30.0, -27.0, 41.0, -10.0)


def test_resolve_country_polygon_unions_multiple_countries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(countries_module, "_country_borders", _fake_borders)

    polygon = resolve_country_polygon(["Mozambique", "United Republic of Tanzania"])

    assert polygon.bounds == (29.0, -27.0, 41.0, -1.0)


def test_resolve_country_polygon_unknown_country_raises_with_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(countries_module, "_country_borders", _fake_borders)

    with pytest.raises(ValueError, match="United Republic of Tanzania"):
        resolve_country_polygon(["Tanzania"])


def test_resolve_country_polygon_unknown_country_no_hint_if_nothing_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(countries_module, "_country_borders", _fake_borders)

    with pytest.raises(ValueError, match=r"Unknown country 'Xyzzyx'\.$"):
        resolve_country_polygon(["Xyzzyx"])


def test_suggest_prefers_unique_substring_match_over_fuzzy_ratio() -> None:
    names = ["United Republic of Tanzania", "Mozambique", "Mauritania"]

    assert _suggest("Tanzania", names) == " Did you mean 'United Republic of Tanzania'?"


def test_suggest_returns_empty_string_when_nothing_close() -> None:
    names = ["United Republic of Tanzania", "Mozambique"]

    assert _suggest("Xyzzyx", names) == ""


@pytest.mark.slow
def test_resolve_country_polygon_live_matches_known_bbox() -> None:
    polygon = resolve_country_polygon(["Mozambique", "United Republic of Tanzania"])

    min_lon, min_lat, max_lon, max_lat = polygon.bounds
    # Mozambique + Tanzania together span roughly this box - loose bounds, not
    # exact, since the dataset's boundary precision may shift slightly over time.
    assert 25 < min_lon < 32
    assert -30 < min_lat < -20
    assert 38 < max_lon < 45
    assert -3 < max_lat < 2
