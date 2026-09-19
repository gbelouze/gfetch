import pytest

from gfetch.sources import SOURCES, StacSource, get_source


def test_get_source_known() -> None:
    source = get_source("earthsearch")
    assert source.name == "earthsearch"
    assert source.api_url.startswith("https://")


def test_get_source_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown source"):
        get_source("not-a-real-source")


def test_registered_sources_resolve_sentinel2_collection() -> None:
    for source in SOURCES.values():
        collection = source.collection("sentinel-2")
        assert isinstance(collection, str)
        assert collection


def test_collection_unknown_satellite() -> None:
    source = StacSource("earthsearch", "https://example.com")
    with pytest.raises(ValueError, match="No known collection"):
        source.collection("not-a-real-satellite")
