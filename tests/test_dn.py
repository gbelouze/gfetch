import datetime
from pathlib import Path

import numpy as np
import pystac
import pytest
import rasterio
from rasterio.transform import from_origin

from gfetch import dn as dn_module
from gfetch.dn import (
    DN_OFFSET,
    DN_OFFSET_FROM_PROPERTY,
    DN_OFFSET_PROPERTY,
    assign_dn_offsets,
    classify_from_pixels,
    dn_shift,
)


def _item(
    date: str = "2024-07-19",
    *,
    baseline: str = "05.11",
    applied: bool | None = None,
    item_id: str = "item",
) -> pystac.Item:
    properties: dict[str, object] = {"s2:processing_baseline": baseline}
    if applied is not None:
        properties["earthsearch:boa_offset_applied"] = applied
    return pystac.Item(
        id=item_id,
        geometry=None,
        bbox=None,
        datetime=datetime.datetime.fromisoformat(date).replace(tzinfo=datetime.UTC),
        properties=properties,
    )


@pytest.mark.parametrize(
    ("source", "collection", "item", "expected"),
    [
        ("earthsearch", "sentinel-2-c1-l2a", _item(baseline="05.00"), DN_OFFSET),
        ("earthsearch", "sentinel-2-l2a", _item("2018-06-10", baseline="00.01"), 0),
        ("earthsearch", "sentinel-2-l2a", _item("2022-06-01", baseline="04.00", applied=True), 0),
        # Contradicts its metadata, which claims the offset is still there.
        ("earthsearch", "sentinel-2-l2a", _item("2025-02-08", applied=False), 0),
        # Inside the window, but Earth Search removed the offset.
        ("earthsearch", "sentinel-2-l2a", _item("2022-02-01", baseline="04.00", applied=True), 0),
        ("planetary-computer", "sentinel-2-l2a", _item("2020-07-24", baseline="02.12"), 0),
        ("planetary-computer", "sentinel-2-l2a", _item("2022-02-01", baseline="04.00"), DN_OFFSET),
    ],
)
def test_assign_dn_offsets_from_metadata(
    source: str, collection: str, item: pystac.Item, expected: int, monkeypatch
) -> None:
    monkeypatch.setattr(dn_module, "classify_from_pixels", _no_pixel_reads)

    assign_dn_offsets([item], source, collection)

    assert item.properties[DN_OFFSET_PROPERTY] == expected
    assert item.properties[DN_OFFSET_FROM_PROPERTY] == "metadata"


def _no_pixel_reads(item: pystac.Item) -> int | None:
    raise AssertionError(f"{item.id}: read its pixels despite unambiguous metadata")


def test_assign_dn_offsets_leaves_unknown_collections_alone() -> None:
    item = _item()

    assign_dn_offsets([item], "earthsearch", "landsat-c2-l2")

    assert DN_OFFSET_PROPERTY not in item.properties


def test_assign_dn_offsets_classifies_the_ambiguous_window_by_pixels_then_date(
    monkeypatch, caplog
) -> None:
    by_pixels = _item("2022-02-01", baseline="04.00", applied=False, item_id="clear")
    by_date = _item("2022-01-25", baseline="04.00", applied=False, item_id="cloudy")
    outside = _item("2022-03-01", baseline="04.00", applied=False, item_id="outside")
    monkeypatch.setattr(
        dn_module, "classify_from_pixels", lambda item: 0 if item.id == "clear" else None
    )

    with caplog.at_level("WARNING"):
        assign_dn_offsets([by_pixels, by_date, outside], "earthsearch", "sentinel-2-l2a")

    assert by_pixels.properties[DN_OFFSET_PROPERTY] == 0
    assert by_pixels.properties[DN_OFFSET_FROM_PROPERTY] == "pixels"
    assert by_date.properties[DN_OFFSET_PROPERTY] == DN_OFFSET
    assert by_date.properties[DN_OFFSET_FROM_PROPERTY] == "date"
    assert outside.properties[DN_OFFSET_FROM_PROPERTY] == "metadata"
    assert "heuristic" in caplog.text


def _write(path: Path, data: np.ndarray) -> pystac.Asset:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:32736",
        transform=from_origin(600000, 9500020, 10, 10),
    ) as dst:
        dst.write(data, 1)
    return pystac.Asset(href=str(path))


def _item_with_bands(tmp_path: Path, blue: int, scl: int) -> pystac.Item:
    item = _item()
    item.add_asset("blue", _write(tmp_path / "blue.tif", np.full((100, 100), blue, np.uint16)))
    item.add_asset("scl", _write(tmp_path / "scl.tif", np.full((100, 100), scl, np.uint8)))
    return item


@pytest.mark.parametrize(
    ("blue", "scl", "expected"),
    [
        (300, 4, 0),
        (1300, 4, DN_OFFSET),
        (1300, 6, DN_OFFSET),
        # Clouds are bright whatever the encoding: no clear dark pixels to tell from.
        (1300, 9, None),
        (300, 0, None),
    ],
)
def test_classify_from_pixels(tmp_path: Path, blue: int, scl: int, expected: int | None) -> None:
    assert classify_from_pixels(_item_with_bands(tmp_path, blue, scl)) == expected


def test_classify_from_pixels_is_inconclusive_when_dark_pixels_straddle_1000(
    tmp_path: Path,
) -> None:
    blue = np.full((100, 100), 1300, np.uint16)
    blue[:20] = 300
    item = _item()
    item.add_asset("blue", _write(tmp_path / "blue.tif", blue))
    item.add_asset("scl", _write(tmp_path / "scl.tif", np.full((100, 100), 4, np.uint8)))

    assert classify_from_pixels(item) is None


def test_classify_from_pixels_is_inconclusive_when_unreadable(tmp_path: Path) -> None:
    item = _item()
    item.add_asset("blue", pystac.Asset(href=str(tmp_path / "missing.tif")))
    item.add_asset("scl", pystac.Asset(href=str(tmp_path / "missing.tif")))

    assert classify_from_pixels(item) is None


@pytest.mark.parametrize(("offset", "expected"), [(0, DN_OFFSET), (DN_OFFSET, 0), (None, 0)])
def test_dn_shift(offset: int | None, expected: int) -> None:
    item = _item()
    if offset is not None:
        item.properties[DN_OFFSET_PROPERTY] = offset

    assert dn_shift(item) == expected
