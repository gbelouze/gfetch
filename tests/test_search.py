import datetime
from dataclasses import dataclass

import pystac
import pystac_client
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


@dataclass
class _FakePage:
    items: list[pystac.Item]


class _FakeResult:
    def __init__(self, items: list[pystac.Item]) -> None:
        self._items = items

    def matched(self) -> int:
        return len(self._items)

    def pages(self):
        yield _FakePage(self._items)


class _FakeCatalog:
    def __init__(
        self, requested_collections: list[list[str]], requested_kwargs: list[dict] | None = None
    ) -> None:
        self._requested = requested_collections
        self._requested_kwargs = requested_kwargs

    def search(self, *, collections, bbox, intersects, datetime, query, max_items):
        self._requested.append(collections)
        if self._requested_kwargs is not None:
            self._requested_kwargs.append({"bbox": bbox, "intersects": intersects})
        return _FakeResult([])


def test_search_uses_explicit_collection_bypassing_source_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unregistered satellite name would make `source.collection()` raise - the
    explicit `collection` must bypass that lookup entirely, never calling it.
    """
    requested_collections: list[list[str]] = []
    monkeypatch.setattr(
        pystac_client.Client,
        "open",
        staticmethod(lambda url: _FakeCatalog(requested_collections)),
    )

    source = get_source("earthsearch")
    search(
        source,
        "not-a-registered-satellite",
        bbox=(2.0, 48.0, 2.1, 48.1),
        datetime="2024-01-01/2024-06-01",
        collection="my-explicit-collection",
    )

    assert requested_collections == [["my-explicit-collection"]]


def test_search_intersects_drops_bbox_from_the_actual_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The STAC API spec treats `bbox`/`intersects` as mutually exclusive - passing
    `intersects` must drop `bbox` from the request `search()` actually sends, even
    though `bbox` is still required as an argument (used for logging).
    """
    requested_kwargs: list[dict] = []
    monkeypatch.setattr(
        pystac_client.Client,
        "open",
        staticmethod(lambda url: _FakeCatalog([], requested_kwargs)),
    )

    source = get_source("earthsearch")
    polygon = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
    search(
        source,
        "sentinel-2",
        bbox=(2.0, 48.0, 2.1, 48.1),
        datetime="2024-01-01/2024-06-01",
        intersects=polygon,
    )

    assert requested_kwargs == [{"bbox": None, "intersects": polygon}]


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


@pytest.mark.parametrize(
    "datetime_range",
    [
        "2022-01-01/2022-09-30",
        "2021-12-15/2022-01-10",
        "2019-06-01/2020-06-01",
        "2021-01-01/..",
    ],
)
def test_search_refuses_known_earthsearch_c1_gaps(
    datetime_range: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("searched despite a known coverage gap")

    monkeypatch.setattr(pystac_client.Client, "open", no_network)

    with pytest.raises(ValueError, match="sentinel-2-c1-l2a"):
        search(get_source("earthsearch"), "sentinel-2", (-1.0, 45.6, 2.0, 47.7), datetime_range)


@pytest.mark.parametrize(
    ("source", "datetime_range"),
    [
        ("earthsearch", "2021-01-01/2021-12-31"),
        ("earthsearch", "2019-12-01/2021-12-31"),
        ("earthsearch", "2023-01-01T00:00:00Z/2023-06-01T00:00:00Z"),
        ("planetary-computer", "2022-01-01/2022-09-30"),
    ],
)
def test_search_allows_time_ranges_outside_known_gaps(
    source: str, datetime_range: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Searched(Exception):
        pass

    def fake_open(*args: object, **kwargs: object) -> None:
        raise _Searched

    monkeypatch.setattr(pystac_client.Client, "open", fake_open)

    with pytest.raises(_Searched):
        search(get_source(source), "sentinel-2", (-1.0, 45.6, 2.0, 47.7), datetime_range)
