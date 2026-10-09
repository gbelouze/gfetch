"""Rewrite a gfetch Zarr store in the layout `mosaic` currently writes.

The source is either stacked (one `(band, y, x)` array, as `mosaic` writes it) or
per-band (one `(y, x)` array per band, as it wrote before 2026-10-07), at any chunk and
shard size. The output is written through the same functions as `mosaic`'s
(`stack_bands`, `prepare_template`, `write_region`), so it holds the same arrays,
attributes and encodings as a store `mosaic` would have written with the same bands,
chunks and shards. `SKIPPED_SHARDS_ATTR` is recomputed for the new storage grid: a new
storage unit is skipped when every source unit it overlaps was.

The unit of work is one storage unit of the new store, so several invocations (e.g. a
SLURM job array) can split a store between them via `task_id`/`n_tasks`, each writing
disjoint files, and an interrupted run resumes where it stopped.
"""

import logging
import shutil
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
from typing import cast

import xarray as xr
import zarr
from rich.progress import Progress
from zarr.core.metadata import ArrayV3Metadata

from gfetch.utils.progress import temporary_task
from gfetch.write import (
    BAND_DIM,
    BAND_NAMES_ATTR,
    SKIPPED_SHARDS_ATTR,
    STACKED_VARIABLE,
    prepare_template,
    region_is_skipped,
    region_is_written,
    stack_bands,
    store_initialized,
    store_is_complete,
    write_region,
    write_regions,
)

log = logging.getLogger(__name__)

__all__ = [
    "has_layout",
    "open_per_band",
    "rechunk",
    "spatial_arrays",
    "stored_bands",
    "write_template",
]

SPATIAL_DIMS = ("y", "x")


def spatial_arrays(path: Path) -> list[str]:
    """
    List a store's arrays that have both a `y` and an `x` dimension.

    Parameters
    ----------
    path : Path
        Zarr store path.

    Returns
    -------
    list[str]
        Array names, sorted.
    """
    group = zarr.open_group(store=path, mode="r")
    names = []
    for name, arr in group.arrays():
        assert isinstance(arr.metadata, ArrayV3Metadata), f"{path / name} is not Zarr v3"
        if set(SPATIAL_DIMS) <= set(arr.metadata.dimension_names or ()):
            names.append(name)
    return sorted(names)


def stored_bands(path: Path, arrays: list[str]) -> list[str] | None:
    """
    Read a stacked store's band names.

    Parameters
    ----------
    path : Path
        Zarr store path.
    arrays : list[str]
        The store's spatial arrays, as listed by `spatial_arrays`.

    Returns
    -------
    list[str] | None
        The band names, in order, or None if the store isn't stacked.
    """
    if arrays != [STACKED_VARIABLE]:
        return None
    arr = zarr.open_array(store=path / STACKED_VARIABLE, mode="r")
    return list(cast("list[str]", arr.attrs.get(BAND_NAMES_ATTR, [])))


def has_layout(path: Path, chunk: int, shard: int) -> bool:
    """
    Check a stacked store's spatial chunk and shard size.

    Parameters
    ----------
    path : Path
        Zarr store path, holding `STACKED_VARIABLE`.
    chunk : int
        Expected chunk side along `y` and `x`, in pixels.
    shard : int
        Expected shard side along `y` and `x`, in pixels; equal to `chunk` for an
        unsharded store.

    Returns
    -------
    bool
        True if both match along `y` and `x`.
    """
    arr = zarr.open_array(store=path / STACKED_VARIABLE, mode="r")
    assert isinstance(arr.metadata, ArrayV3Metadata)
    dims = list(arr.metadata.dimension_names or ())
    chunks = dict(zip(dims, arr.chunks, strict=True))
    shards = dict(zip(dims, arr.shards or arr.chunks, strict=True))
    return all(chunks[d] == chunk and shards[d] == shard for d in SPATIAL_DIMS)


def open_per_band(path: Path) -> xr.Dataset:
    """
    Open a stacked or per-band store lazily, as one `(y, x)` variable per band.

    Encodings are dropped, and the grid mapping is set as a coordinate, so the dataset
    looks like the mosaic `mosaic` hands to `stack_bands`.

    Parameters
    ----------
    path : Path
        Zarr store path.

    Returns
    -------
    xr.Dataset
        One lazily loaded `(y, x)` variable per band, with the store's root
        attributes but `SKIPPED_SHARDS_ATTR`.
    """
    ds = xr.open_zarr(path, consolidated=False, chunks=None).drop_encoding()
    attrs = {k: v for k, v in ds.attrs.items() if k != SKIPPED_SHARDS_ATTR}
    # The grid mapping reopens as a data variable; `mosaic` carries it as a coordinate.
    ds = ds.set_coords([v for v in ds.data_vars if not set(SPATIAL_DIMS) <= set(ds[v].dims)])
    if STACKED_VARIABLE in ds.data_vars:
        ds = ds[STACKED_VARIABLE].to_dataset(dim=BAND_DIM).drop_vars(BAND_DIM, errors="ignore")
    ds.attrs = attrs
    return ds


def write_template(
    source: xr.Dataset,
    dst: Path,
    bands: Sequence[str],
    chunk: int,
    shard: int,
    skip: Callable[[dict[str, slice]], bool],
) -> None:
    """
    Write the template `mosaic` would write for `source`'s grid, bands and attributes.

    Parameters
    ----------
    source : xr.Dataset
        Dataset as opened by `open_per_band`.
    dst : Path
        Destination Zarr store path.
    bands : Sequence[str]
        Bands to stack, in order.
    chunk : int
        Chunk side along `y` and `x`, in pixels.
    shard : int
        Shard side along `y` and `x`, in pixels; equal to `chunk` for an unsharded
        store.
    skip : Callable[[dict[str, slice]], bool]
        Storage units to record as skipped, see `gfetch.write.prepare_template`.
    """
    prepare_template(
        stack_bands(source.chunk(dict.fromkeys(SPATIAL_DIMS, shard)), bands),
        dst,
        shards=dict.fromkeys(SPATIAL_DIMS, shard) if shard != chunk else None,
        chunks=dict.fromkeys(SPATIAL_DIMS, chunk),
        skip=skip,
    )


def _copy_region(
    source: xr.Dataset, dst: Path, bands: Sequence[str], region: dict[str, slice]
) -> None:
    window = source[list(bands)].isel({d: region[d] for d in SPATIAL_DIMS}).load()
    write_region(stack_bands(window, bands), dst, region)


def _swap(path: Path, new: Path, old: Path) -> None:
    """
    Move `new` onto `path`, through `old`; resumable if interrupted at any step.

    Concurrent callers are safe: each rename succeeds for one of them only, and a
    caller that loses a race leaves the rest to the winner.
    """
    if path.exists():
        try:
            path.rename(old)
        except FileNotFoundError:
            return
    try:
        new.rename(path)
    except FileNotFoundError:
        return
    shutil.rmtree(old, ignore_errors=True)
    log.info(f"Replaced {path} with its rewritten copy")


def rechunk(
    path: Path,
    chunk: int,
    shard: int,
    *,
    bands: Sequence[str] | None = None,
    output: Path | None = None,
    task_id: int = 0,
    n_tasks: int = 1,
    n_workers: int = 1,
    progress: Progress | None = None,
) -> None:
    """
    Rewrite a Zarr store with stacked bands and a new chunk and shard size.

    Parameters
    ----------
    path : Path
        Zarr store to rewrite, stacked or per-band. Must be complete (see
        `gfetch.write.store_is_complete`).
    chunk : int
        Chunk side along `y` and `x`, in pixels.
    shard : int
        Shard side along `y` and `x`, in pixels, a multiple of `chunk`. Equal to
        `chunk` for an unsharded store.
    bands : Sequence[str] | None
        Band order of the new store: `mosaic`'s order for the job, i.e. the config's
        `bands` (each followed by its `_ascending`/`_descending` variant with
        `orbit_state: as_bands`). Must name exactly the store's bands. Defaults to None,
        which keeps a stacked store's order, and is refused for a per-band store, whose
        order isn't recorded.
    output : Path | None
        Where to write the new store. Defaults to None, which builds it next to `path`
        as `.<name>.rechunk` and moves it onto `path` once complete.
    task_id : int
        This invocation's index among `n_tasks` concurrent invocations. Defaults to 0.
    n_tasks : int
        Total number of concurrent invocations splitting the store's units between
        them. Defaults to 1 (no splitting).
    n_workers : int
        Number of threads copying units. Each holds about three units' worth of every
        band in memory. Defaults to 1.
    progress : Progress | None
        Progress bar to report copied units on. Defaults to None (no progress bar).

    Raises
    ------
    ValueError
        If `shard` isn't a multiple of `chunk`; if `path` isn't a complete store or has
        no array with both a `y` and an `x` dimension; if `bands` is None for a
        per-band store, or doesn't name exactly the store's bands; or if a partly
        written new store has another layout.
    """
    if shard % chunk:
        raise ValueError(f"Shard side {shard} is not a multiple of chunk side {chunk}.")
    in_place = output is None
    dst = output if output is not None else path.parent / f".{path.name}.rechunk"
    old = path.parent / f".{path.name}.rechunk-old"
    if in_place and not path.exists() and old.exists():
        log.info(f"Resuming an interrupted swap of {path}")
        _swap(path, dst, old)
        return

    arrays = spatial_arrays(path)
    if not arrays:
        raise ValueError(f"{path} has no array with both a 'y' and an 'x' dimension.")
    if not store_is_complete(path, arrays):
        raise ValueError(f"{path} is not completely written, finish writing it first.")
    stored = stored_bands(path, arrays)
    available = stored if stored is not None else arrays
    if bands is None:
        if stored is None:
            raise ValueError(
                f"{path} has one array per band, whose order isn't recorded: pass the "
                f"bands in mosaic's order (found {available})."
            )
        bands = stored
    if sorted(bands) != sorted(available):
        raise ValueError(f"Bands {list(bands)} don't match {path}'s {available}.")
    bands = list(bands)
    if in_place and stored == bands and has_layout(path, chunk, shard):
        log.info(f"{path} already has the target layout, skipping")
        return

    source = open_per_band(path)
    if not store_initialized(dst):
        skip_var = arrays[0]
        write_template(
            source,
            dst,
            bands,
            chunk,
            shard,
            skip=lambda region: region_is_skipped(path, skip_var, region),
        )
    if stored_bands(dst, spatial_arrays(dst)) != bands or not has_layout(dst, chunk, shard):
        raise ValueError(
            f"{dst} already exists with other bands or another layout than {chunk} px "
            f"chunks and {shard} px shards; delete it first."
        )

    regions = write_regions(dst, STACKED_VARIABLE)[task_id::n_tasks]
    todo = [r for r in regions if not region_is_written(dst, r, [STACKED_VARIABLE])]
    log.info(
        f"{path}: {len(todo)} unit(s) to write, {len(regions) - len(todo)} already written "
        f"(task {task_id}/{n_tasks})"
    )
    with (
        (
            temporary_task(progress, f"Rewriting {path.name}", total=len(todo))
            if progress is not None
            else nullcontext()
        ) as task,
        ThreadPoolExecutor(n_workers) as pool,
    ):
        futures = {pool.submit(_copy_region, source, dst, bands, r): r for r in todo}
        for i, future in enumerate(as_completed(futures), start=1):
            future.result()
            region = futures[future]
            log.info(
                f"{path.name} unit (y={region['y'].start}, x={region['x'].start}): "
                f"written [{i}/{len(todo)}]"
            )
            if progress is not None and task is not None:
                progress.advance(task)

    if in_place and store_is_complete(dst, [STACKED_VARIABLE]):
        _swap(path, dst, old)
