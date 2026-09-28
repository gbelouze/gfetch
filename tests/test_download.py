import asyncio
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import aiohttp
import pystac
import pytest
from multidict import CIMultiDict, CIMultiDictProxy
from rich.progress import Progress
from stac_asset import DownloadError
from yarl import URL

from gfetch.download import _s3_uri_to_public_https, download_items


def _make_item(item_id: str, assets: dict[str, Path | str]) -> pystac.Item:
    item = pystac.Item(
        id=item_id,
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]},
        bbox=[0, 0, 1, 1],
        datetime=datetime.now(UTC),
        properties={},
    )
    for key, path in assets.items():
        item.add_asset(key, pystac.Asset(href=str(path)))
    return item


def test_download_items_basic(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    red_src = source_dir / "red.tif"
    red_src.write_bytes(b"RED-BYTES")
    scl_src = source_dir / "scl.tif"
    scl_src.write_bytes(b"SCL-BYTES")

    item = _make_item("item-1", {"red": red_src, "scl": scl_src})
    cache_dir = tmp_path / "cache"

    [result] = asyncio.run(download_items([item], cache_dir, ["red", "scl"]))

    item_dir = cache_dir / "item-1"
    assert (item_dir / "red.tif").read_bytes() == b"RED-BYTES"
    assert (item_dir / "scl.tif").read_bytes() == b"SCL-BYTES"
    assert (item_dir / "red.complete").exists()
    assert (item_dir / "scl.complete").exists()

    assert result.assets["red"].href == str(item_dir / "red.tif")
    assert result.assets["scl"].href == str(item_dir / "scl.tif")
    assert result.get_self_href() == str(item_dir / "item-1.json")


def test_download_items_restricts_to_requested_keys(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    red_src = source_dir / "red.tif"
    red_src.write_bytes(b"RED-BYTES")
    nir_src = source_dir / "nir.tif"
    nir_src.write_bytes(b"NIR-BYTES")

    item = _make_item("item-1", {"red": red_src, "nir": nir_src})
    cache_dir = tmp_path / "cache"

    [result] = asyncio.run(download_items([item], cache_dir, ["red"]))

    assert set(result.assets) == {"red"}
    assert not (cache_dir / "item-1" / "nir.tif").exists()


def test_download_items_missing_key_is_skipped(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    red_src = source_dir / "red.tif"
    red_src.write_bytes(b"RED-BYTES")

    item = _make_item("item-1", {"red": red_src})
    cache_dir = tmp_path / "cache"

    [result] = asyncio.run(download_items([item], cache_dir, ["red", "does-not-exist"]))

    assert set(result.assets) == {"red"}


def test_download_items_resume_skips_completed_assets(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    red_src = source_dir / "red.tif"
    red_src.write_bytes(b"RED-BYTES")

    item = _make_item("item-1", {"red": red_src})
    cache_dir = tmp_path / "cache"

    asyncio.run(download_items([item], cache_dir, ["red"]))

    # The source becomes unavailable after the first download: a resumed run must
    # succeed without needing it again.
    red_src.unlink()

    [result] = asyncio.run(download_items([item], cache_dir, ["red"]))

    item_dir = cache_dir / "item-1"
    assert (item_dir / "red.tif").read_bytes() == b"RED-BYTES"
    assert result.assets["red"].href == str(item_dir / "red.tif")


def test_download_items_retry_does_not_corrupt_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed attempt must not leave the item in a state (relative hrefs, a stale
    self href) that breaks the next retry - download_item() mutates its input item in
    place, so each attempt must operate on a fresh copy.
    """
    import gfetch.download as download_mod

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    red_src = source_dir / "red.tif"
    red_src.write_bytes(b"RED-BYTES")

    item = _make_item("item-1", {"red": red_src})
    cache_dir = tmp_path / "cache"

    real_download_item = download_mod.download_item
    call_count = 0

    async def flaky_download_item(item, directory, *, config, keep_non_downloaded):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise TimeoutError("simulated transient failure")
        return await real_download_item(
            item, directory, config=config, keep_non_downloaded=keep_non_downloaded
        )

    monkeypatch.setattr(download_mod, "download_item", flaky_download_item)

    [result] = asyncio.run(download_items([item], cache_dir, ["red"]))

    assert call_count == 2
    item_dir = cache_dir / "item-1"
    assert (item_dir / "red.tif").read_bytes() == b"RED-BYTES"
    assert result.assets["red"].href == str(item_dir / "red.tif")


def test_download_items_reports_progress(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    items = []
    for i in range(3):
        src = source_dir / f"red-{i}.tif"
        src.write_bytes(f"RED-{i}".encode() * 1000)
        items.append(_make_item(f"item-{i}", {"red": src}))

    cache_dir = tmp_path / "cache"
    progress = Progress(disable=True)

    with mock.patch.object(progress, "add_task", wraps=progress.add_task) as add_task:
        results = asyncio.run(
            download_items(items, cache_dir, ["red"], max_concurrent_items=2, progress=progress)
        )

    assert len(results) == 3
    for i, result in enumerate(results):
        item_dir = cache_dir / f"item-{i}"
        assert (item_dir / "red.tif").read_bytes() == f"RED-{i}".encode() * 1000
        assert result.assets["red"].href == str(item_dir / "red.tif")

    # One task for the main "Downloading items" bar, plus one per item's own
    # byte-progress bar.
    assert add_task.call_count == 1 + len(items)
    # ...and every one of them was cleaned up (temporary_task's finally block) once
    # it finished; none linger after the call returns.
    assert len(progress.tasks) == 0


@pytest.mark.parametrize(
    ("href", "expected"),
    [
        (
            "s3://sentinel-s1-l1c/GRD/2026/6/15/iw-vv.tiff",
            "https://sentinel-s1-l1c.s3.amazonaws.com/GRD/2026/6/15/iw-vv.tiff",
        ),
        # already public HTTPS - stac_asset's HttpClient, not its region-guessing
        # S3Client, so nothing to rewrite
        (
            "https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/B04.tif",
            "https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/B04.tif",
        ),
        ("/local/cache/item-1/red.tif", "/local/cache/item-1/red.tif"),
    ],
)
def test_s3_uri_to_public_https(href: str, expected: str) -> None:
    assert _s3_uri_to_public_https(href) == expected


def test_download_items_rewrites_s3_hrefs_before_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """download_item() should never see a raw s3:// href - it must already be rewritten
    to the region-agnostic public HTTPS form (see _s3_uri_to_public_https).
    """
    import gfetch.download as download_mod

    item = _make_item("item-1", {"vv": "s3://sentinel-s1-l1c/GRD/iw-vv.tiff"})
    cache_dir = tmp_path / "cache"

    seen_hrefs: list[str] = []

    async def fake_download_item(item, directory, *, config, keep_non_downloaded, **kwargs):
        seen_hrefs.append(item.assets["vv"].href)
        (directory / "iw-vv.tiff").write_bytes(b"VV-BYTES")
        item.assets["vv"].href = "iw-vv.tiff"
        item.set_self_href(str(directory / "item-1.json"))
        return item

    monkeypatch.setattr(download_mod, "download_item", fake_download_item)

    asyncio.run(download_items([item], cache_dir, ["vv"]))

    assert seen_hrefs == ["https://sentinel-s1-l1c.s3.amazonaws.com/GRD/iw-vv.tiff"]


def test_download_items_multiple_items(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    items = []
    for i in range(3):
        src = source_dir / f"red-{i}.tif"
        src.write_bytes(f"RED-{i}".encode())
        items.append(_make_item(f"item-{i}", {"red": src}))

    cache_dir = tmp_path / "cache"
    results = asyncio.run(download_items(items, cache_dir, ["red"], max_concurrent_items=2))

    assert len(results) == 3
    for i, result in enumerate(results):
        item_dir = cache_dir / f"item-{i}"
        assert (item_dir / "red.tif").read_bytes() == f"RED-{i}".encode()
        assert result.assets["red"].href == str(item_dir / "red.tif")


def _http_error(href: str, status: int) -> aiohttp.ClientResponseError:
    url = URL(f"{href}?sig=token")
    request_info = aiohttp.RequestInfo(url, "GET", CIMultiDictProxy(CIMultiDict()), url)
    return aiohttp.ClientResponseError(request_info, (), status=status)


def _failing_download_item(
    monkeypatch: pytest.MonkeyPatch, failures: dict[str, list[int]]
) -> list[list[str]]:
    """Patch download_item() to fail on the assets in `failures`, each popping its next
    HTTP status per call it's included in, and return the asset keys of every call.
    """
    import gfetch.download as download_mod

    real_download_item = download_mod.download_item
    calls: list[list[str]] = []

    async def fake_download_item(item, directory, *, config, keep_non_downloaded):
        calls.append(list(config.include))
        errors: list[Exception] = [
            _http_error(item.assets[key].href, failures[key].pop(0))
            for key in config.include
            if failures.get(key)
        ]
        if errors:
            raise DownloadError(errors)
        return await real_download_item(
            item, directory, config=config, keep_non_downloaded=keep_non_downloaded
        )

    monkeypatch.setattr(download_mod, "download_item", fake_download_item)
    return calls


def test_download_items_skips_missing_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for key in ("vv", "vh"):
        (source_dir / f"{key}.tif").write_bytes(key.encode())
    item = _make_item("item-1", {k: source_dir / f"{k}.tif" for k in ("vv", "vh")})
    calls = _failing_download_item(monkeypatch, {"vh": [404]})
    cache_dir = tmp_path / "cache"

    [result] = asyncio.run(download_items([item], cache_dir, ["vv", "vh"]))

    assert calls == [["vv", "vh"], ["vv"]]
    item_dir = cache_dir / "item-1"
    assert list(result.assets) == ["vv"]
    assert result.assets["vv"].href == str(item_dir / "vv.tif")
    assert (item_dir / "vv.complete").exists()
    assert not (item_dir / "vh.complete").exists()
    assert "skipping asset(s) ['vh']" in caplog.text


def test_download_items_drops_item_with_all_assets_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    items = []
    for i in range(2):
        src = source_dir / f"vv-{i}.tif"
        src.write_bytes(f"VV-{i}".encode())
        items.append(_make_item(f"item-{i}", {"vv": src}))
    _failing_download_item(monkeypatch, {"vv": [404]})

    results = asyncio.run(download_items(items, tmp_path / "cache", ["vv"]))

    assert [r.id for r in results] == ["item-1"]


def test_download_items_retries_other_http_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    src = source_dir / "vv.tif"
    src.write_bytes(b"VV")
    item = _make_item("item-1", {"vv": src})
    calls = _failing_download_item(monkeypatch, {"vv": [500]})

    [result] = asyncio.run(download_items([item], tmp_path / "cache", ["vv"]))

    assert calls == [["vv"], ["vv"]]
    assert list(result.assets) == ["vv"]
