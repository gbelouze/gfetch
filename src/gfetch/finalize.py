"""Post-mosaic finalization: pack finished Zarr stores into single-file zip stores, and
remove the download cache once every store it feeds is complete.

Both exist to lower inode usage on HPC filesystems with per-user inode quotas: a Zarr
store holds one file per chunk per variable, and the download cache several files per
item. See `claude/tech-stack.md`'s "Inode usage" section.
"""

import logging
import os
import shutil
import tempfile
import zipfile
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

import zarr
from rich.progress import Progress
from zarr.storage import ZipStore

from gfetch.utils.progress import temporary_task
from gfetch.write import store_is_complete

log = logging.getLogger(__name__)

__all__ = ["pack_store", "packed_store_path", "remove_cache"]


def packed_store_path(store: Path) -> Path:
    """
    Path of the zip store `pack_store` writes for `store`.

    Parameters
    ----------
    store : Path
        Zarr store directory, e.g. `mosaic_epsg32631.zarr`.

    Returns
    -------
    Path
        Sibling zip path, e.g. `mosaic_epsg32631.zarr.zip`.
    """
    return store.with_name(f"{store.name}.zip")


def pack_store(
    store: Path,
    variables: Sequence[str],
    *,
    remove_source: bool = False,
    progress: Progress | None = None,
) -> Path:
    """
    Pack a complete Zarr store directory into a single uncompressed zip file.

    The result is a regular Zarr store, readable in place via
    `zarr.storage.ZipStore(path, mode="r")` (e.g. `xr.open_zarr(ZipStore(...))`) or
    GDAL's `/vsizip/` prefix, and costs one inode instead of one per chunk. It is
    read-only: a zip can't be written to concurrently, so pack only once `mosaic` is
    done with the store.

    Chunks are stored as-is, not recompressed, since Zarr already compresses them.
    Files are streamed into the archive one at a time, so memory use doesn't grow
    with store size. The archive is built under a temporary name next to its
    destination and renamed into place only once complete, so the zip path existing
    means packing finished; an interrupted run leaves a hidden `.<name>.*.tmp` file
    behind and is simply rerun. Rerunning after a successful pack is a no-op (apart
    from `remove_source`).

    Parameters
    ----------
    store : Path
        Zarr store directory to pack.
    variables : Sequence[str]
        Data variables (e.g. the mosaic's bands) that must be fully written before
        packing is allowed, see `gfetch.write.store_is_complete`.
    remove_source : bool
        Delete `store` once its zip is in place. Defaults to False.
    progress : Progress | None
        Rich progress tracker, advanced by bytes packed. Defaults to None (no
        progress bar).

    Returns
    -------
    Path
        Path of the zip store, see `packed_store_path`.

    Raises
    ------
    FileNotFoundError
        If neither `store` nor its zip exist.
    ValueError
        If `store` isn't complete.
    """
    dest = packed_store_path(store)
    if dest.exists():
        log.info(f"{dest} already exists, skipping packing")
    else:
        if not store.exists():
            raise FileNotFoundError(f"{store} not found")
        if not store_is_complete(store, variables):
            msg = f"{store} is incomplete (missing chunks for {list(variables)}), not packing it"
            log.error(msg)
            raise ValueError(msg)
        _zip_directory(store, dest, progress)
        log.info(f"Packed {store} into {dest}")

    if remove_source and store.exists():
        shutil.rmtree(store)
        log.info(f"Removed {store}")
    return dest


def _zip_directory(src: Path, dest: Path, progress: Progress | None) -> None:
    """
    Atomically write `src`'s files into an uncompressed zip at `dest`.

    Parameters
    ----------
    src : Path
        Directory to pack; its hidden files (e.g. leftover temporary files) are
        skipped.
    dest : Path
        Destination zip path, overwritten if present.
    progress : Progress | None
        Rich progress tracker, advanced by bytes packed.
    """
    files = sorted(
        p
        for p in src.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(src).parts)
    )
    total_bytes = sum(p.stat().st_size for p in files)
    log.debug(f"Packing {len(files)} file(s), {total_bytes} byte(s) from {src}")

    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with (
            os.fdopen(fd, "wb") as f,
            temporary_task(progress, f"Packing {src.name}", total=total_bytes)
            if progress is not None
            else nullcontext(None) as task,
        ):
            with zipfile.ZipFile(f, "w", compression=zipfile.ZIP_STORED) as zf:
                for p in files:
                    zf.write(p, p.relative_to(src).as_posix())
                    if progress is not None and task is not None:
                        progress.advance(task, p.stat().st_size)
            f.flush()
            os.fsync(f.fileno())
        # Fails before the rename if the archive isn't a readable Zarr store.
        zarr.open_group(store=ZipStore(tmp, mode="r"), mode="r")
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def remove_cache(
    cache_dir: Path,
    stores: Sequence[Path],
    variables: Sequence[str],
    *,
    cached_items_path: Path | None = None,
    progress: Progress | None = None,
) -> None:
    """
    Delete the download cache, once every store built from it is complete.

    A store counts as complete if it has been packed (see `pack_store`, which only
    packs complete stores) or if `gfetch.write.store_is_complete` says so. Nothing
    is deleted unless every store in `stores` is complete.

    `cached_items_path`, whose asset hrefs point into `cache_dir`, is deleted first,
    so that an interrupted removal never leaves it pointing at a half-deleted cache;
    rerunning finishes the removal.

    Parameters
    ----------
    cache_dir : Path
        Download cache directory, one subdirectory per item.
    stores : Sequence[Path]
        Every Zarr store built from `cache_dir` (e.g. one per UTM zone).
    variables : Sequence[str]
        Data variables (e.g. the mosaic's bands) each store must have fully written.
    cached_items_path : Path | None
        Hand-off file listing the cached items, deleted along with the cache.
        Defaults to None (no such file).
    progress : Progress | None
        Rich progress tracker, advanced once per removed cache entry. Defaults to
        None (no progress bar).

    Raises
    ------
    ValueError
        If `stores` is empty, or any store in it isn't complete.
    """
    if not stores:
        raise ValueError("No stores given, refusing to remove the cache unchecked")
    incomplete = [
        s
        for s in stores
        if not packed_store_path(s).exists() and not store_is_complete(s, variables)
    ]
    if incomplete:
        msg = f"Store(s) {[str(s) for s in incomplete]} incomplete, not removing {cache_dir}"
        log.error(msg)
        raise ValueError(msg)

    if cached_items_path is not None:
        cached_items_path.unlink(missing_ok=True)
    if not cache_dir.exists():
        log.info(f"{cache_dir} already removed")
        return

    entries = sorted(cache_dir.iterdir())
    with (
        temporary_task(progress, "Removing cache", total=len(entries))
        if progress is not None
        else nullcontext(None)
    ) as task:
        for entry in entries:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            if progress is not None and task is not None:
                progress.advance(task)
    cache_dir.rmdir()
    log.info(f"Removed {len(entries)} cache entries from {cache_dir}")
