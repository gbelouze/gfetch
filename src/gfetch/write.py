"""Write stage: persist a mosaic to a local Zarr store.

Default output target is plain Zarr (not Icechunk) with pre-planned, non-overlapping
per-worker chunk regions, safe on any POSIX filesystem, including shared HPC storage.
See `claude/tech-stack.md`'s "HPC / distributed execution" section for why.
"""

import asyncio
import errno
import itertools
import logging
import shutil
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import xarray as xr
import zarr
from zarr.core.metadata import ArrayV3Metadata
from zarr.storage import StoreLike

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
    "write_regions",
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


def validate_chunks(
    path: Path,
    variables: Sequence[str],
    expected_chunks: dict[str, int],
    expected_shards: dict[str, int] | None = None,
) -> None:
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
    any shard is built - a cheap metadata read per variable, no chunk data touched.

    Parameters
    ----------
    path : Path
        Zarr store path, already initialized via `prepare_template`.
    variables : Sequence[str]
        Data variables to check.
    expected_chunks : dict[str, int]
        Expected chunk size per dimension name (e.g. `{"y": 256, "x": 256}`), as
        resolved by `gfetch.mosaic.resolve_chunks`. With sharding, these are the
        inner chunks.
    expected_shards : dict[str, int] | None
        Expected shard size per dimension name, as resolved by
        `gfetch.mosaic.resolve_shards`. Defaults to None, which expects an unsharded
        store.

    Raises
    ------
    ValueError
        If any variable's actual on-disk chunk or shard size differs from
        `expected_chunks`/`expected_shards` for a dimension it has.
    """
    for var in variables:
        za = zarr.open_array(store=path / var)
        assert isinstance(za.metadata, ArrayV3Metadata), f"{path / var} is not Zarr v3"
        dims = za.metadata.dimension_names
        assert dims is not None, f"{path / var} has no dimension names"
        # An unsharded store's write unit is its chunk.
        expected_write = expected_shards if expected_shards is not None else expected_chunks
        for label, actual_sizes, expected_sizes in (
            ("chunk", za.chunks, expected_chunks),
            ("shard", za.shards or za.chunks, expected_write),
        ):
            actual = dict(zip(dims, actual_sizes, strict=True))
            for dim, expected in expected_sizes.items():
                if actual.get(dim) is not None and actual[dim] != expected:
                    raise ValueError(
                        f"{path / var}: on-disk {label} size for dimension {dim!r} is "
                        f"{actual[dim]}, but this run's configuration expects {expected}. "
                        "The store was likely created by an earlier run under a different "
                        f"configuration - delete {path} and let it be recreated, or fix "
                        "the configuration to match the existing store."
                    )


def prepare_template(
    ds: xr.Dataset,
    path: Path,
    shards: dict[str, int] | None = None,
    chunks: dict[str, int] | None = None,
) -> None:
    """
    Write only a Zarr store's metadata and coordinates, without any chunk data.

    Coordinating step for a pre-planned disjoint-region write: call this before
    writing any region, so the store's shape/coords/chunking exist first. Idempotent
    and safe to call from every worker unconditionally, including concurrently: the
    template is written into a private temporary directory next to `path`, then
    renamed onto `path` in one atomic `rename`, so `path` is either absent or a
    complete template. `rename` refuses to replace a non-empty directory, so the
    first caller's template wins and every other caller discards its own. This only
    works because every caller derives an identical template from the same
    deterministic inputs (geobox, bands, chunks), so whichever caller's write actually
    lands is immaterial.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset whose shape/coords/chunking define the store; its data isn't written.
        When `shards` and `chunks` are given, its dask chunks only need to align with
        the shards, so building it shard-sized keeps its graph small.
    path : Path
        Destination Zarr store path.
    shards : dict[str, int] | None
        Shard size per dimension name, each a multiple of that dimension's chunk
        size. Dimensions not listed get one chunk per shard. Defaults to None (no
        sharding).
    chunks : dict[str, int] | None
        Store chunk size per dimension name (the inner chunks, when sharded).
        Dimensions not listed keep `ds`'s dask chunking. Defaults to None, which uses
        `ds`'s dask chunks.
    """
    ds = _restore_grid_mapping(ds)
    encoding = {}
    if shards is None:
        if chunks is not None:
            ds = ds.chunk({dim: size for dim, size in chunks.items() if dim in ds.dims})
    else:
        # The inner chunks go straight into the encoding, never through `ds.chunk`:
        # dask builds a rechunk's tasks eagerly, one per output chunk, which over a
        # whole UTM zone at small chunk sizes is tens of millions of tasks.
        store_chunks = chunks if chunks is not None else {}
        for name, var in ds.data_vars.items():
            dask_chunks = tuple(c[0] for c in var.chunks) if var.chunks is not None else var.shape
            var_chunks = tuple(
                store_chunks.get(str(dim), chunk)
                for dim, chunk in zip(var.dims, dask_chunks, strict=True)
            )
            encoding[name] = {
                "chunks": var_chunks,
                "shards": tuple(
                    shards.get(str(dim), chunk)
                    for dim, chunk in zip(var.dims, var_chunks, strict=True)
                ),
            }
        # xarray requires dask chunks aligned to shards; nothing is computed here.
        ds = ds.chunk({dim: size for dim, size in shards.items() if dim in ds.dims})
    if store_initialized(path):
        log.debug(f"{path} already initialized, skipping template write")
        return
    tmp_path = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        ds.to_zarr(tmp_path, compute=False, mode="w-", encoding=encoding)
        try:
            tmp_path.rename(path)
        except OSError as e:
            if e.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                raise
            log.debug(f"{path} initialized concurrently, discarding this template")
            return
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
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
        e.g. `{"x": slice(0, 512), "y": slice(None)}`, typically one of
        `write_regions`.
    """
    ds = _restore_grid_mapping(ds)
    if ds.chunks:
        # A shard is written by a single task, so its dask chunks are merged first.
        # Computing stays at the original, smaller chunk size.
        ds = ds.chunk({dim: -1 for dim in region if dim in ds.dims})
    # xarray's region-write rejects any variable lacking a dimension in common with
    # `region`; scalar coordinates like a CRS grid mapping variable must be dropped,
    # since they were already written once by prepare_template.
    ds = ds.drop_vars([c for c in ds.coords if c not in ds.dims])
    # zarr-python skips writing chunks that are entirely fill value (e.g. an all-NaN
    # nodata shard) by default, which `region_is_written` couldn't tell apart from a
    # chunk that was never written.
    ds.to_zarr(path, region=region, write_empty_chunks=True)
    log.debug(f"Wrote region {region} to {path}")


def write_regions(store: StoreLike, variable: str) -> list[dict[str, slice]]:
    """
    List a variable's storage units (shards, or chunks if unsharded) as regions.

    Each region is one independent unit of work for `write_region`: two writers never
    touch the same file as long as each writes whole regions from this list.

    Parameters
    ----------
    store : StoreLike
        Zarr store holding `variable`, e.g. a store directory `Path` or a read-only
        `zarr.storage.ZipStore`.
    variable : str
        Array whose storage grid to list.

    Returns
    -------
    list[dict[str, slice]]
        Mapping from dimension name to slice for every unit, in row-major order,
        edge units clipped to the array's extent.
    """
    za = zarr.open_group(store=store, mode="r")[variable]
    assert isinstance(za, zarr.Array), f"{variable} is not an array"
    assert isinstance(za.metadata, ArrayV3Metadata), f"{variable} is not Zarr v3"
    assert za.metadata.dimension_names is not None, f"{variable} has no dimension names"
    dims = [dim for dim in za.metadata.dimension_names if dim is not None]
    assert len(dims) == za.ndim, f"{variable} has an unnamed dimension"
    per_dim_slices = []
    for sizes in za.write_chunk_sizes:
        bounds = [0, *itertools.accumulate(sizes)]
        per_dim_slices.append([slice(a, b) for a, b in itertools.pairwise(bounds)])
    return [dict(zip(dims, slices, strict=True)) for slices in itertools.product(*per_dim_slices)]


def region_is_written(path: Path, region: dict[str, slice], variables: Sequence[str]) -> bool:
    """
    Check whether every storage unit covering `region` already exists on disk.

    A storage unit is a shard, or a chunk in an unsharded store. Reads their
    boundaries and dimension order from the store's own array metadata rather than
    taking them as a parameter, so this can't drift from whatever `prepare_template`
    actually wrote. A unit's file existing means it was fully written: zarr-python's
    `LocalStore` writes every file atomically (temp file + rename), and `write_region`
    always writes whole units, so there's no separate partial-write state to guard
    against here.

    Parameters
    ----------
    path : Path
        Zarr store path, already initialized via `prepare_template`.
    region : dict[str, slice]
        Mapping from dimension name to the region to check, as passed to
        `write_region`. A dimension missing from `region` is checked in full.
    variables : Sequence[str]
        Data variables to check; the region only counts as written once every listed
        variable's covering units are all present.

    Returns
    -------
    bool
        True if every unit covering `region` already exists for every variable in
        `variables`.
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
            write_sizes = za.shards or za.chunks
            for dim, chunk_size, size in zip(dims, write_sizes, za.shape, strict=True):
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
