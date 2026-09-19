import pytest

from gfetch.search import search
from gfetch.sources import get_source


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
