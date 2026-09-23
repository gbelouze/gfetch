"""Write stage: persist a mosaic to a local Zarr store.

Default output target is plain Zarr (not Icechunk) with pre-planned, non-overlapping
per-worker chunk regions, safe on any POSIX filesystem, including shared HPC storage.
See `claude/tech-stack.md`'s "HPC / distributed execution" section for why.
"""

import asyncio
import itertools
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import xarray as xr
import zarr
from zarr.core.metadata import ArrayV3Metadata

from gfetch.utils.memory import log_chunk_footprint

log = logging.getLogger(__name__)

__all__ = [
    "prepare_template",
    "region_is_written",
    "store_initialized",
    "store_is_complete",
    "validate_chunks",
    "write",
    "write_region",
]

_ZarrMode = Literal["a", "a-", "r", "r+", "w", "w-"]


def _restore_grid_mapping(ds: xr.Dataset) -> xr.Dataset:
    """
    Re-attach each data variable to its CRS coordinate via `.encoding["grid_mapping"]`.

    `odc.stac.load` sets this link through `.encoding`, not `.attrs`; any subsequent
    computation (`.where`, `.median`, arithmetic, ...) drops `.encoding` on its output,
    since xarray treats it as serialization-only metadata rather than data to
    propagate. Left unrestored, the link never reaches `to_zarr`, so the on-disk CRS
    coordinate (still fully CF-compliant on its own) ends up orphaned: readers that
    fall back to scanning for a coordinate with `spatial_ref`/`crs_wkt` attributes
    (xarray, odc-geo, rioxarray) still find it, but strict CF readers like GDAL's Zarr
    driver report the array as unreferenced.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset about to be persisted.

    Returns
    -------
    xr.Dataset
        `ds` with `grid_mapping` encoding restored on every data variable, or `ds`
        itself unchanged if no CRS coordinate is present.
    """
    crs_coords = [
        name
        for name, coord in ds.coords.items()
        if coord.ndim == 0 and ("spatial_ref" in coord.attrs or "crs_wkt" in coord.attrs)
    ]
    if not crs_coords:
        return ds
    (crs_coord,) = crs_coords
    ds = ds.copy()
    for var in ds.data_vars.values():
        var.encoding.setdefault("grid_mapping", crs_coord)
    return ds


def write(ds: xr.Dataset, path: Path, *, mode: _ZarrMode = "w") -> None:
    """
    Write a dataset to a local Zarr store in one call.

    Use `prepare_template`/`write_region` instead when multiple workers need to write
    disjoint regions of the same store independently.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to write, e.g. a mosaic from `gfetch.mosaic.mosaic`.
    path : Path
        Destination Zarr store path.
    mode : _ZarrMode
        `xarray.Dataset.to_zarr` write mode. Defaults to 'w' (overwrite).
    """
    ds = _restore_grid_mapping(ds)
    log.debug(f"Writing dataset {dict(ds.sizes)} to {path} (mode={mode})")
    log_chunk_footprint(ds, log)
    ds.to_zarr(path, mode=mode, write_empty_chunks=True)
    log.info(f"Wrote dataset to {path}")


def store_initialized(path: Path) -> bool:
    """
    Check whether `prepare_template` has already been run against `path`.

    A cheap existence check (no store open, no metadata parsing) meant to let a
    caller skip building an expensive template dataset (e.g. `mosaic()` over a whole
    zone's items) before ever finding out `prepare_template` would have been a
    no-op. `prepare_template` stays idempotent and safe to call unconditionally on
    its own (see its docstring); this is purely an optimization on top; a real race
    against a concurrently-initializing task just means this returns False once and
    `prepare_template` deduplicates as it already does.

    Parameters
    ----------
    path : Path
        Zarr store path.

    Returns
    -------
    bool
        True if `path` already holds a Zarr v3 root group.
    """
    return (path / "zarr.json").exists()


def validate_chunks(path: Path, variables: Sequence[str], expected_chunks: dict[str, int]) -> None:
    """
    Check that an already-initialized store's on-disk chunk grid matches what this
    run expects, failing fast and clearly instead of letting a mismatch surface deep
    inside a later `write_region`/`to_zarr(region=...)` call.

    A store's chunk grid is fixed forever once `prepare_template` first writes it -
    if it was created by an earlier run under a different chunk configuration (or a
    different gfetch version, or briefly, before 2026-09-22, a run whose own build
    wasn't guaranteed to chunk every variable identically - see `gfetch.mosaic.
    mosaic`'s docstring), every later write against it will keep failing the same
    way until the mismatch is noticed and the store is recreated. Call this once per
    store, right after confirming it already exists (`store_initialized`), before
    any patch is built - a cheap metadata read per variable, no chunk data touched.

    Parameters
    ----------
    path : Path
        Zarr store path, already initialized via `prepare_template`.
    variables : Sequence[str]
        Data variables to check.
    expected_chunks : dict[str, int]
        Expected chunk size per dimension name (e.g. `{"y": 2048, "x": 2048}`), as
        resolved by `gfetch.mosaic.resolve_chunks`.

    Raises
    ------
    ValueError
        If any variable's actual on-disk chunk size differs from `expected_chunks`
        for a dimension it has.
    """
    for var in variables:
        za = zarr.open_array(store=path / var)
        assert isinstance(za.metadata, ArrayV3Metadata), f"{path / var} is not Zarr v3"
        dims = za.metadata.dimension_names
        assert dims is not None, f"{path / var} has no dimension names"
        actual_chunks = dict(zip(dims, za.chunks, strict=True))
        for dim, expected in expected_chunks.items():
            actual = actual_chunks.get(dim)
            if actual is not None and actual != expected:
                raise ValueError(
                    f"{path / var}: on-disk chunk size for dimension {dim!r} is "
                    f"{actual}, but this run's configuration expects {expected}. The "
                    "store was likely created by an earlier run under a different "
                    f"configuration - delete {path} and let it be recreated, or fix "
                    "the configuration to match the existing store."
                )


def prepare_template(ds: xr.Dataset, path: Path) -> None:
    """
    Write only a Zarr store's metadata and coordinates, without any chunk data.

    Coordinating step for a pre-planned disjoint-region write: call this before
    writing any region, so the store's shape/coords/chunking exist first. Idempotent
    and safe to call from every worker unconditionally, including concurrently:
    it's a no-op if `path` is already initialized, relying on `mode="w-"`
    failing atomically at the filesystem level if the store already exists. This only
    works because every caller derives an identical template from the same
    deterministic inputs (geobox, bands, chunks), so whichever caller's write actually
    lands is immaterial.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset whose shape/coords/chunking define the store; its data isn't written.
    path : Path
        Destination Zarr store path.
    """
    ds = _restore_grid_mapping(ds)
    try:
        ds.to_zarr(path, compute=False, mode="w-")
    except FileExistsError:
        log.debug(f"{path} already initialized, skipping template write")
        return
    log.info(f"Wrote Zarr template (metadata only) to {path}")


def write_region(ds: xr.Dataset, path: Path, region: dict[str, slice]) -> None:
    """
    Write one disjoint region of an already-templated Zarr store.

    Parameters
    ----------
    ds : xr.Dataset
        The slice of the full dataset this worker is responsible for; must align
        exactly with `region` and with the store's chunk boundaries (the standard
        Dask-to-Zarr region-write requirement).
    path : Path
        Zarr store path, already initialized via `prepare_template`.
    region : dict[str, slice]
        Mapping from dimension name to the slice of the store this call writes,
        e.g. `{"x": slice(0, 512), "y": slice(None)}`.
    """
    ds = _restore_grid_mapping(ds)
    # xarray's region-write rejects any variable lacking a dimension in common with
    # `region`; scalar coordinates like a CRS grid mapping variable must be dropped,
    # since they were already written once by prepare_template.
    ds = ds.drop_vars([c for c in ds.coords if c not in ds.dims])
    # zarr-python skips writing chunks that are entirely fill value (e.g. an all-NaN
    # nodata patch) by default, which `region_is_written` couldn't tell apart from a
    # chunk that was never written.
    ds.to_zarr(path, region=region, write_empty_chunks=True)
    log.debug(f"Wrote region {region} to {path}")


def region_is_written(path: Path, region: dict[str, slice], variables: Sequence[str]) -> bool:
    """
    Check whether every native Zarr chunk covering `region` already exists on disk.

    Reads chunk boundaries and dimension order from the store's own array metadata
    rather than taking them as a parameter, so this can't drift from whatever
    `prepare_template` actually wrote. A chunk file existing means it was fully
    written, not partially: zarr-python's `LocalStore` writes every chunk file
    atomically (temp file + rename), so there's no separate partial-write state to
    guard against here.

    Parameters
    ----------
    path : Path
        Zarr store path, already initialized via `prepare_template`.
    region : dict[str, slice]
        Mapping from dimension name to the region to check, as passed to
        `write_region`. A dimension missing from `region` is checked in full.
    variables : Sequence[str]
        Data variables to check; the region only counts as written once every listed
        variable's covering chunks are all present.

    Returns
    -------
    bool
        True if every native chunk covering `region` already exists for every
        variable in `variables`.
    """

    async def _all_written() -> bool:
        checks = []
        for var in variables:
            za = zarr.open_array(store=path / var)
            # gfetch only ever writes Zarr v3 (`prepare_template`'s `to_zarr` default);
            # v2 arrays have no `dimension_names` to key `region` off of.
            assert isinstance(za.metadata, ArrayV3Metadata), f"{path / var} is not Zarr v3"
            dims = za.metadata.dimension_names
            assert dims is not None, f"{path / var} has no dimension names"
            axis_chunk_ranges = []
            for dim, chunk_size, size in zip(dims, za.chunks, za.shape, strict=True):
                assert dim is not None, f"{path / var} has an unnamed dimension"
                sl = region.get(dim, slice(None))
                start = sl.start if sl.start is not None else 0
                stop = sl.stop if sl.stop is not None else size
                axis_chunk_ranges.append(range(start // chunk_size, -(-stop // chunk_size)))
            for coords in itertools.product(*axis_chunk_ranges):
                checks.append(za.store.exists(za.metadata.encode_chunk_key(coords)))
        return all(await asyncio.gather(*checks))

    return asyncio.run(_all_written())


def store_is_complete(path: Path, variables: Sequence[str]) -> bool:
    """
    Check whether every chunk of every variable in `variables` has been written.

    Parameters
    ----------
    path : Path
        Zarr store path.
    variables : Sequence[str]
        Data variables to check (e.g. the mosaic's bands).

    Returns
    -------
    bool
        True if `path` is an initialized store whose `variables` are fully written.
    """
    return store_initialized(path) and region_is_written(path, {}, variables)
