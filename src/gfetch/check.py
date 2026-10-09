"""Check that a Zarr store has the form `gfetch mosaic` writes.

The reference is `mosaic`'s own writer: the store's metadata is compared, file by file,
with the template `mosaic` would write for the store's grid, bands, chunks and shards
(see `gfetch.rechunk.write_template`), so the checks can't drift from what `mosaic`
actually writes. A few properties the template can't vouch for, because it is built from
the store itself (bands, layout, georeferencing, skipped shards, completeness), are
checked separately.
"""

import json
import logging
import math
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import zarr

from gfetch.rechunk import (
    SPATIAL_DIMS,
    has_layout,
    open_per_band,
    spatial_arrays,
    stored_bands,
    write_template,
)
from gfetch.write import (
    SKIPPED_SHARDS_ATTR,
    STACKED_VARIABLE,
    region_is_written,
    write_regions,
)

log = logging.getLogger(__name__)

__all__ = ["CheckResult", "check_store"]


@dataclass(frozen=True)
class CheckResult:
    """
    Outcome of one check of a store.

    Attributes
    ----------
    name : str
        What was checked.
    ok : bool
        Whether the store passed.
    detail : str
        What was found, mostly useful when the check failed.
    """

    name: str
    ok: bool
    detail: str = ""


def _metadata(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(p.relative_to(path)): json.loads(p.read_text())
        for p in sorted(path.rglob("zarr.json"))
        if not any(part.startswith(".") for part in p.relative_to(path).parts)
    }


def _metadata_diff(path: Path, reference: Path) -> list[str]:
    actual, expected = _metadata(path), _metadata(reference)
    diffs = [f"missing {name}" for name in sorted(expected.keys() - actual.keys())]
    diffs += [f"unexpected {name}" for name in sorted(actual.keys() - expected.keys())]
    for name in sorted(actual.keys() & expected.keys()):
        a, e = actual[name], expected[name]
        if name == "zarr.json":
            a, e = (
                {
                    **m,
                    "attributes": {
                        k: v for k, v in m.get("attributes", {}).items() if k != SKIPPED_SHARDS_ATTR
                    },
                }
                for m in (a, e)
            )
        for key in sorted(a.keys() | e.keys()):
            if a.get(key) != e.get(key):
                diffs.append(f"{name}: {key} is {a.get(key)!r}, expected {e.get(key)!r}")
    return diffs


@contextmanager
def _quiet(logger_name: str) -> Iterator[None]:
    """Hide a logger's INFO records, e.g. those of writing the reference template."""
    logger = logging.getLogger(logger_name)
    level = logger.level
    logger.setLevel(max(level, logging.WARNING))
    try:
        yield
    finally:
        logger.setLevel(level)


def _check_skipped(path: Path) -> CheckResult:
    name = f"{SKIPPED_SHARDS_ATTR} attribute"
    record = zarr.open_group(store=path, mode="r").attrs.get(SKIPPED_SHARDS_ATTR)
    if not isinstance(record, dict):
        return CheckResult(name, False, f"missing or not a mapping: {record!r}")
    if record.get("dimensions") != list(SPATIAL_DIMS):
        return CheckResult(name, False, f"dimensions are {record.get('dimensions')!r}")
    arr = zarr.open_array(store=path / STACKED_VARIABLE, mode="r")
    units = arr.shards or arr.chunks
    grid = [math.ceil(arr.shape[-2 + i] / units[-2 + i]) for i in range(2)]
    indices = record.get("indices")
    if not isinstance(indices, list):
        return CheckResult(name, False, f"indices are {indices!r}")
    bad = [
        i
        for i in indices
        if not (
            isinstance(i, list)
            and len(i) == 2
            and all(isinstance(v, int) and 0 <= v < n for v, n in zip(i, grid, strict=True))
        )
    ]
    if bad:
        return CheckResult(
            name, False, f"{len(bad)} index(es) outside the {grid} grid, e.g. {bad[0]}"
        )
    return CheckResult(name, True, f"{len(indices)} skipped shard(s)")


def _check_georeferenced(path: Path) -> CheckResult:
    name = "georeferenced"
    group = zarr.open_group(store=path, mode="r")
    grid_mapping = group[STACKED_VARIABLE].attrs.get("grid_mapping")
    if not isinstance(grid_mapping, str) or grid_mapping not in group:
        return CheckResult(name, False, f"grid_mapping {grid_mapping!r} names no array")
    attrs = group[grid_mapping].attrs
    if "crs_wkt" not in attrs:
        return CheckResult(name, False, f"{grid_mapping} has no crs_wkt attribute")
    return CheckResult(name, True, f"grid mapping {grid_mapping!r}")


def check_store(
    path: Path, chunk: int, shard: int, bands: Sequence[str] | None = None
) -> list[CheckResult]:
    """
    Check that a Zarr store has the form `gfetch mosaic` writes.

    Checks run in order and stop at the first failure the later ones depend on.

    Parameters
    ----------
    path : Path
        Zarr store path.
    chunk : int
        Expected chunk side along `y` and `x`, in pixels.
    shard : int
        Expected shard side along `y` and `x`, in pixels; equal to `chunk` for an
        unsharded store.
    bands : Sequence[str] | None
        Expected band names, in order. Defaults to None, which accepts any.

    Returns
    -------
    list[CheckResult]
        One result per check run.
    """
    results: list[CheckResult] = []
    root = path / "zarr.json"
    if not root.exists():
        return [CheckResult("Zarr v3 group", False, f"no {root}")]
    meta = json.loads(root.read_text())
    if meta.get("zarr_format") != 3 or meta.get("node_type") != "group":
        return [CheckResult("Zarr v3 group", False, f"{root} isn't a Zarr v3 group")]
    results.append(CheckResult("Zarr v3 group", True))
    consolidated = meta.get("consolidated_metadata") is not None
    results.append(
        CheckResult(
            "unconsolidated metadata",
            not consolidated,
            "consolidated metadata makes GDAL drop the CRS" if consolidated else "",
        )
    )

    arrays = spatial_arrays(path)
    stored = stored_bands(path, arrays)
    if not stored:
        detail = (
            f"one array per band {arrays}: run `gfetch utils rechunk --bands ...`"
            if stored is None
            else f"{STACKED_VARIABLE} has no band names"
        )
        return [*results, CheckResult("stacked bands", False, detail)]
    results.append(CheckResult("stacked bands", True, f"{stored}"))
    if bands is not None:
        results.append(
            CheckResult(
                "band order",
                list(bands) == stored,
                "" if list(bands) == stored else f"{stored}, expected {list(bands)}",
            )
        )

    layout_ok = has_layout(path, chunk, shard)
    arr = zarr.open_array(store=path / STACKED_VARIABLE, mode="r")
    results.append(
        CheckResult(
            "chunk and shard size",
            layout_ok,
            f"chunks {arr.chunks}, shards {arr.shards}"
            + ("" if layout_ok else f", expected {chunk} px chunks and {shard} px shards"),
        )
    )
    georeferenced = _check_georeferenced(path)
    results.append(georeferenced)

    if georeferenced.ok:
        # At the store's own layout, so that a layout mismatch is only reported above.
        own_chunk, own_shard = arr.chunks[-1], (arr.shards or arr.chunks)[-1]
        with tempfile.TemporaryDirectory() as tmp, _quiet("gfetch.write"):
            reference = Path(tmp) / path.name
            write_template(
                open_per_band(path),
                reference,
                stored,
                own_chunk,
                own_shard,
                skip=lambda region: False,
            )
            diffs = _metadata_diff(path, reference)
        results.append(
            CheckResult(
                "metadata as mosaic writes it",
                not diffs,
                "; ".join(diffs[:5]) + (f" (+{len(diffs) - 5} more)" if len(diffs) > 5 else ""),
            )
        )

    skipped = _check_skipped(path)
    results.append(skipped)
    if skipped.ok:
        regions = write_regions(path, STACKED_VARIABLE)
        missing = sum(not region_is_written(path, r, [STACKED_VARIABLE]) for r in regions)
        results.append(
            CheckResult(
                "completely written",
                not missing,
                f"{missing}/{len(regions)} shard(s) unwritten"
                if missing
                else f"{len(regions)} shard(s)",
            )
        )
    return results
