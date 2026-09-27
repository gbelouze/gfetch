"""Post-mosaic finalization: remove the download cache once every store it feeds is
complete.

Lowers inode usage on HPC filesystems with per-user inode quotas: the download cache
holds several files per item. See `claude/tech-stack.md`'s "Inode usage" section.
"""

import logging
import shutil
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

from rich.progress import Progress

from gfetch.utils.progress import temporary_task
from gfetch.write import store_is_complete

log = logging.getLogger(__name__)

__all__ = ["remove_cache"]


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

    Nothing is deleted unless `gfetch.write.store_is_complete` holds for every store
    in `stores`.

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
    incomplete = [s for s in stores if not store_is_complete(s, variables)]
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
