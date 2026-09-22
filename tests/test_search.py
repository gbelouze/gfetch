import datetime

import pystac
import pytest

from gfetch.search import _dedupe_sentinel2_processing_baseline, search
from gfetch.sources import get_source


def _s2_item(item_id: str, *, grid_code: str, date: str, baseline: str) -> pystac.Item:
    return pystac.Item(
        id=item_id,
        geometry=None,
        bbox=None,
        datetime=datetime.datetime.fromisoformat(date).replace(tzinfo=datetime.UTC),
        properties={"grid:code": grid_code, "s2:processing_baseline": baseline},
    )


def test_dedupe_sentinel2_processing_baseline_keeps_highest_baseline() -> None:
    old = _s2_item("old", grid_code="MGRS-35MNM", date="2020-06-06", baseline="02.14")
    new = _s2_item("new", grid_code="MGRS-35MNM", date="2020-06-06", baseline="05.00")

    deduped = _dedupe_sentinel2_processing_baseline([old, new])

    assert [item.id for item in deduped] == ["new"]


def test_dedupe_sentinel2_processing_baseline_keeps_distinct_tiles_and_dates() -> None:
    tile_a = _s2_item("a", grid_code="MGRS-35MNM", date="2020-06-06", baseline="02.14")
    tile_b = _s2_item("b", grid_code="MGRS-35MNN", date="2020-06-06", baseline="02.14")
    other_date = _s2_item("c", grid_code="MGRS-35MNM", date="2020-06-16", baseline="02.14")

    deduped = _dedupe_sentinel2_processing_baseline([tile_a, tile_b, other_date])

    assert {item.id for item in deduped} == {"a", "b", "c"}


@pytest.mark.slow
def test_search_earthsearch_returns_items() -> None:
    source = get_source("earthsearch")
    items = search(
        source,
        "sentinel-2",
        bbox=(2.30, 48.85, 2.35, 48.90),
        datetime="2026-06-01/2026-06-30",
        query={"eo:cloud_cover": {"lt": 40}},
    )

    assert items
    for item in items:
        assert "red" in item.assets
        assert "scl" in item.assets


@pytest.mark.slow
def test_search_earthsearch_sentinel1_returns_items() -> None:
    source = get_source("earthsearch")
    items = search(
        source,
        "sentinel-1",
        bbox=(2.0, 48.6, 2.6, 49.0),
        datetime="2026-06-01/2026-06-15",
    )

    assert items
    for item in items:
        assert "vv" in item.assets
        assert "vh" in item.assets
        assert item.properties.get("sat:orbit_state") in {"ascending", "descending"}


@pytest.mark.slow
def test_search_earthsearch_sentinel1_orbit_state_filter() -> None:
    source = get_source("earthsearch")
    items = search(
        source,
        "sentinel-1",
        bbox=(2.0, 48.6, 2.6, 49.0),
        datetime="2026-05-01/2026-06-15",
        query={"sat:orbit_state": {"eq": "descending"}},
    )

    assert items
    assert {item.properties.get("sat:orbit_state") for item in items} == {"descending"}
