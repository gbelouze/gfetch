"""Sentinel-2 L2A DN encoding: which items carry the +1000 reflectance offset.

Since processing baseline 04.00, ESA stores `DN = reflectance * 10000 + 1000`. Collection
1 does throughout, but not every item a source serves does, and the STAC metadata doesn't
always say which (see `claude/s2-2022/README.md`). `assign_dn_offsets` records each
item's offset as `DN_OFFSET_PROPERTY`, and `gfetch.mosaic`/`gfetch.ocm` shift every item to
`DN_OFFSET` before using its pixels.
"""

import datetime
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pystac
import rasterio
from rasterio.enums import Resampling

__all__ = [
    "DN_OFFSET",
    "DN_OFFSET_FROM_PROPERTY",
    "DN_OFFSET_PROPERTY",
    "REFLECTANCE_BANDS",
    "assign_dn_offsets",
    "classify_from_pixels",
    "dn_shift",
]

log = logging.getLogger(__name__)

DN_OFFSET = 1000
DN_OFFSET_PROPERTY = "gfetch:dn_offset"
DN_OFFSET_FROM_PROPERTY = "gfetch:dn_offset_from"

# AOT, WVP, SCL and the 8-bit `visual` asset have their own encodings, without the offset.
REFLECTANCE_BANDS = frozenset(
    {
        "coastal",
        "blue",
        "green",
        "red",
        "rededge1",
        "rededge2",
        "rededge3",
        "nir",
        "nir08",
        "nir09",
        "swir16",
        "swir22",
    }
)

# Earth Search v1 `sentinel-2-l2a` items acquired then, with `boa_offset_applied: false`,
# mostly carry the offset; every other item of that collection doesn't, whatever its
# metadata says. Measured 2026-10-16 over four regions, `claude/s2-2022/README.md`.
_AMBIGUOUS_WINDOW = (datetime.date(2022, 1, 25), datetime.date(2022, 2, 27))

_COLLECTIONS = frozenset(
    {
        ("earthsearch", "sentinel-2-c1-l2a"),
        ("earthsearch", "sentinel-2-l2a"),
        ("planetary-computer", "sentinel-2-l2a"),
    }
)

# Vegetation and water: clear and dark in blue (reflectance ~0.01-0.05), so their DNs sit
# below 1000 without the offset and above it with.
_DARK_SCL = (4, 6)
_CLASSIFY_SHAPE = (549, 549)
_MIN_DARK_PIXELS = 2000
_WITH_OFFSET_MAX_FRACTION = 0.02
_WITHOUT_OFFSET_MIN_FRACTION = 0.5
_CLASSIFY_WORKERS = 8


def _baseline_at_least_04(item: pystac.Item) -> bool:
    return item.properties.get("s2:processing_baseline", "00.00") >= "04.00"


def _in_ambiguous_window(item: pystac.Item) -> bool:
    assert item.datetime is not None
    return (
        item.properties.get("earthsearch:boa_offset_applied") is False
        and _baseline_at_least_04(item)
        and _AMBIGUOUS_WINDOW[0] <= item.datetime.date() <= _AMBIGUOUS_WINDOW[1]
    )


def _metadata_offset(source: str, collection: str, item: pystac.Item) -> int | None:
    """
    Decide an item's DN offset from its source, collection and metadata alone.

    Parameters
    ----------
    source : str
        Source name, e.g. 'earthsearch'.
    collection : str
        Collection id, one of `_COLLECTIONS` for `source`.
    item : pystac.Item
        Item to decide for.

    Returns
    -------
    int | None
        0 or `DN_OFFSET`, or None if only its pixels can tell.
    """
    if (source, collection) == ("earthsearch", "sentinel-2-c1-l2a"):
        return DN_OFFSET
    if (source, collection) == ("earthsearch", "sentinel-2-l2a"):
        return None if _in_ambiguous_window(item) else 0
    return DN_OFFSET if _baseline_at_least_04(item) else 0


def classify_from_pixels(item: pystac.Item) -> int | None:
    """
    Tell an item's DN offset from its blue band over clear vegetation and water.

    Parameters
    ----------
    item : pystac.Item
        Item with readable `blue` and `scl` assets.

    Returns
    -------
    int | None
        0 or `DN_OFFSET`, or None if the item has too few clear dark pixels, or they
        don't clearly fall on either side of 1000, or its assets can't be read.
    """
    try:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES"):
            with rasterio.open(item.assets["blue"].get_absolute_href()) as src:
                blue = src.read(1, out_shape=_CLASSIFY_SHAPE, resampling=Resampling.nearest)
            with rasterio.open(item.assets["scl"].get_absolute_href()) as src:
                scl = src.read(1, out_shape=_CLASSIFY_SHAPE, resampling=Resampling.nearest)
    except (rasterio.errors.RasterioError, KeyError) as e:
        log.warning(f"{item.id}: can't read its blue/scl assets to classify its DN offset: {e}")
        return None
    dark = blue[np.isin(scl, _DARK_SCL) & (blue > 0)]
    if dark.size < _MIN_DARK_PIXELS:
        return None
    below = float((dark < DN_OFFSET).mean())
    if below <= _WITH_OFFSET_MAX_FRACTION:
        return DN_OFFSET
    if below >= _WITHOUT_OFFSET_MIN_FRACTION:
        return 0
    return None


def assign_dn_offsets(items: Sequence[pystac.Item], source: str, collection: str) -> None:
    """
    Set each item's `DN_OFFSET_PROPERTY` and `DN_OFFSET_FROM_PROPERTY`, in place.

    Items of a collection with no known DN encoding (e.g. a `custom` satellite's) are
    left unchanged. Items whose metadata can't tell are classified from their pixels
    (`classify_from_pixels`), and those still undecided are assumed to carry the offset,
    as most items acquired in their window do. Logs a warning whenever that heuristic
    is used.

    Parameters
    ----------
    items : Sequence[pystac.Item]
        Items of `collection`, with gfetch's asset keys (see `gfetch.sources.
        asset_renames`).
    source : str
        Source name the items were searched on.
    collection : str
        Collection id the items belong to.
    """
    if (source, collection) not in _COLLECTIONS:
        log.debug(f"No known DN encoding for {source} {collection!r}, not harmonizing it")
        return
    ambiguous = []
    for item in items:
        offset = _metadata_offset(source, collection, item)
        if offset is None:
            ambiguous.append(item)
        else:
            item.properties[DN_OFFSET_PROPERTY] = offset
            item.properties[DN_OFFSET_FROM_PROPERTY] = "metadata"
    if not ambiguous:
        return

    log.info(f"Classifying the DN offset of {len(ambiguous)} item(s) from their pixels")
    with ThreadPoolExecutor(_CLASSIFY_WORKERS) as pool:
        offsets = list(pool.map(classify_from_pixels, ambiguous))
    for item, offset in zip(ambiguous, offsets, strict=True):
        item.properties[DN_OFFSET_PROPERTY] = DN_OFFSET if offset is None else offset
        item.properties[DN_OFFSET_FROM_PROPERTY] = "date" if offset is None else "pixels"
    by_date = offsets.count(None)
    log.warning(
        f"!!! {len(ambiguous)} item(s) acquired {_AMBIGUOUS_WINDOW[0]} to "
        f"{_AMBIGUOUS_WINDOW[1]} have an ambiguous DN encoding: harmonizing them with a "
        f"heuristic. {len(ambiguous) - by_date} classified from their pixels "
        f"({offsets.count(DN_OFFSET)} with the +{DN_OFFSET} offset, {offsets.count(0)} "
        f"without), {by_date} assumed to carry it by date alone."
    )


def dn_shift(item: pystac.Item) -> int:
    """
    The DN shift bringing an item's reflectance bands to the `DN_OFFSET` encoding.

    Parameters
    ----------
    item : pystac.Item
        Item, with or without `DN_OFFSET_PROPERTY`.

    Returns
    -------
    int
        `DN_OFFSET` minus the item's offset, 0 if it has no `DN_OFFSET_PROPERTY`.
    """
    offset = item.properties.get(DN_OFFSET_PROPERTY)
    return 0 if offset is None else DN_OFFSET - int(offset)
