import pytest

from gfetch.sources import SOURCES, StacSource, get_source, planetary_computer_signer


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


def test_registered_sources_resolve_sentinel1_collection() -> None:
    assert SOURCES["earthsearch"].collection("sentinel-1") == "sentinel-1-grd"
    assert SOURCES["planetary-computer"].collection("sentinel-1") == "sentinel-1-rtc"


def test_collection_unknown_satellite() -> None:
    source = StacSource("earthsearch", "https://example.com")
    with pytest.raises(ValueError, match="No known collection"):
        source.collection("not-a-real-satellite")


def test_planetary_computer_signer_signs_blob_hrefs_once_per_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[str, str]] = []

    def fake_token(account: str, container: str) -> str:
        requests.append((account, container))
        return f"sig={len(requests)}"

    monkeypatch.setattr("gfetch.sources._pc_sas_token", fake_token)
    base = "https://acct.blob.core.windows.net"

    sign = planetary_computer_signer()
    signed = [sign(f"{base}/c1/a.tif"), sign(f"{base}/c1/b.tif"), sign(f"{base}/c2/a.tif")]

    assert signed == [f"{base}/c1/a.tif?sig=1", f"{base}/c1/b.tif?sig=1", f"{base}/c2/a.tif?sig=2"]
    assert planetary_computer_signer()(f"{base}/c1/a.tif") == f"{base}/c1/a.tif?sig=3"
    assert requests == [("acct", "c1"), ("acct", "c2"), ("acct", "c1")]


@pytest.mark.parametrize(
    "href",
    [
        "s3://bucket/a.tif",
        "https://example.com/a.tif",
        "/local/cache/a.tif",
        "https://acct.blob.core.windows.net/c1/a.tif?st=x&se=y&sig=z",
    ],
)
def test_planetary_computer_signer_leaves_other_hrefs_unchanged(
    monkeypatch: pytest.MonkeyPatch, href: str
) -> None:
    def fail(account: str, container: str) -> str:
        raise AssertionError("no token should be requested")

    monkeypatch.setattr("gfetch.sources._pc_sas_token", fail)

    assert planetary_computer_signer()(href) == href
