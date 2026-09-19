import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pystac
import pytest

from gfetch.download import download_items


def _make_item(item_id: str, assets: dict[str, Path]) -> pystac.Item:
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
