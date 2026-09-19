"""Download stage: fetch STAC item assets to a local cache, atomically and resumably.

Downloads land at ``<cache_dir>/<item_id>/<asset_key><suffix>``. Each completed asset
is marked by a sibling ``<cache_dir>/<item_id>/<asset_key>.complete`` sentinel file, so
a killed/preempted job can resume by skipping assets whose sentinel already exists,
without needing a central manifest or any locking - many workers can share one cache
directory safely. Writes are atomic: each item downloads into a temp directory next to
its final location, and only completed files are moved into place.
"""

import asyncio
import copy
import dataclasses
import logging
import shutil
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from contextlib import nullcontext
from pathlib import Path
from urllib.parse import urlsplit

import pystac
from stac_asset import Config, download_item

from gfetch.utils.progress import Progress, TaskID, temporary_task

log = logging.getLogger(__name__)

__all__ = ["download_items"]

_RETRY_TRIES = 5
_RETRY_INITIAL_DELAY = 1.0
_RETRY_BACKOFF = 1.7


async def _retry_async[T](
    coro_fn: Callable[[], Awaitable[T]],
    *,
    tries: int = _RETRY_TRIES,
    delay: float = _RETRY_INITIAL_DELAY,
    backoff: float = _RETRY_BACKOFF,
) -> T:
    """
    Retry an async callable with exponential backoff.

    The synchronous `retry` package used elsewhere in gfetch's sibling projects only
    wraps regular callables - it never awaits a coroutine, so it can't retry
    stac-asset's async downloads. This is a small async-aware equivalent, with the
    same tries/backoff defaults.

    Parameters
    ----------
    coro_fn : Callable[[], Awaitable[T]]
        Zero-argument callable returning an awaitable to retry.
    tries : int
        Maximum number of attempts. Defaults to 5.
    delay : float
        Initial delay in seconds between attempts. Defaults to 1.0.
    backoff : float
        Multiplier applied to the delay after each failed attempt. Defaults to 1.7.

    Returns
    -------
    T
        The result of the first successful attempt.
    """
    attempt = 0
    while True:
        try:
            return await coro_fn()
        except Exception as e:
            attempt += 1
            if attempt >= tries:
                raise
            log.warning(f"Attempt {attempt}/{tries} failed ({e}), retrying in {delay:.1f}s")
            await asyncio.sleep(delay)
            delay *= backoff


async def _download_item(
    item: pystac.Item,
    cache_dir: Path,
    asset_keys: Sequence[str],
    config: Config,
    semaphore: asyncio.Semaphore,
) -> pystac.Item:
    """
    Download the requested assets of one STAC item into `cache_dir`, atomically and
    resumably.

    Parameters
    ----------
    item : pystac.Item
        Item to download; its assets' hrefs are the remote sources.
    cache_dir : Path
        Root cache directory; assets land under `cache_dir/<item.id>/`.
    asset_keys : Sequence[str]
        Asset keys to download. Keys not present on `item` are silently skipped.
    config : Config
        stac-asset download configuration (auth, timeouts, ...). Its `include` field
        is overridden per call to restrict downloads to the pending asset subset.
    semaphore : asyncio.Semaphore
        Bounds the number of items downloading concurrently across the whole batch.

    Returns
    -------
    pystac.Item
        A copy of `item` whose requested asset hrefs point at the local cache instead
        of the remote source, ready to be handed to `odc.stac.load()`.
    """
    item_dir = cache_dir / item.id
    item_dir.mkdir(parents=True, exist_ok=True)

    available_keys = [k for k in asset_keys if k in item.assets]
    pending_keys = [k for k in available_keys if not (item_dir / f"{k}.complete").exists()]

    if pending_keys:
        download_config = dataclasses.replace(config, include=pending_keys)

        async def _attempt() -> pystac.Item:
            # The move out of the temp dir must happen before it's cleaned up on
            # exiting this `with` block, so it stays inside `_attempt` (retried as a
            # whole) rather than after it. A fresh deep copy of `item` is required
            # per attempt too: download_item() mutates its input item in place (sets
            # its self href, rewrites asset hrefs), so retrying against the same
            # object would hand a failed attempt's corrupted state to the next one.
            with tempfile.TemporaryDirectory(dir=item_dir) as tmp:
                downloaded = await download_item(
                    copy.deepcopy(item), Path(tmp), config=download_config, keep_non_downloaded=True
                )
                for key in pending_keys:
                    # .get_absolute_href() (not .href) is required here:
                    # download_item() always relative-ifies asset hrefs once it
                    # sets a self href, which it always does when writing into a
                    # directory.
                    href = downloaded.assets[key].get_absolute_href()
                    assert href is not None, f"downloaded asset {key} has no absolute href"
                    tmp_path = Path(href)
                    final_path = item_dir / f"{key}{tmp_path.suffix}"
                    shutil.move(tmp_path, final_path)
                    (item_dir / f"{key}.complete").touch()
                    downloaded.assets[key].href = str(final_path)
                    log.debug(f"Downloaded {item.id}/{key} -> {final_path}")
            return downloaded

        async with semaphore:
            result = await _retry_async(_attempt)
    else:
        log.debug(f"{item.id}: all requested assets already cached, skipping download")
        result = copy.deepcopy(item)

    for key in available_keys:
        if key not in pending_keys:
            suffix = Path(urlsplit(item.assets[key].href).path).suffix
            result.assets[key].href = str(item_dir / f"{key}{suffix}")

    result.set_self_href(str(item_dir / f"{item.id}.json"))
    result.assets = {k: result.assets[k] for k in available_keys}
    return result


async def download_items(
    items: Sequence[pystac.Item],
    cache_dir: Path,
    asset_keys: Sequence[str],
    *,
    config: Config | None = None,
    max_concurrent_items: int = 4,
    progress: Progress | None = None,
) -> list[pystac.Item]:
    """
    Download a batch of STAC items' assets into a local cache.

    Safe for many callers/processes to share one `cache_dir` concurrently: each asset
    is written atomically (temp file + rename) and marked complete with its own
    sentinel file, so no locking or central manifest is needed, and a killed/resumed
    run just re-checks sentinels rather than re-downloading everything.

    Parameters
    ----------
    items : Sequence[pystac.Item]
        Items to download, as returned by `gfetch.search.search`.
    cache_dir : Path
        Root cache directory; each item's assets land under `cache_dir/<item.id>/`.
    asset_keys : Sequence[str]
        Asset keys to download for every item (e.g. a satellite profile's default
        bands plus its cloud-mask band). Keys absent from a given item are skipped.
    config : Config | None
        stac-asset download configuration (auth, timeouts, ...). Defaults to None,
        which uses `stac_asset.Config()`.
    max_concurrent_items : int
        Maximum number of items downloading concurrently; each item's own assets are
        additionally downloaded concurrently by stac-asset internally. Defaults to 4.
    progress : Progress | None
        Rich progress tracker, advanced once per completed item. Defaults to None.

    Returns
    -------
    list[pystac.Item]
        Copies of `items` whose requested asset hrefs point at the local cache, in
        the same order as `items`, ready to be handed to `odc.stac.load()`.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = config or Config()
    semaphore = asyncio.Semaphore(max_concurrent_items)

    async def _download_one(item: pystac.Item, task: TaskID | None) -> pystac.Item:
        result = await _download_item(item, cache_dir, asset_keys, config, semaphore)
        if progress is not None and task is not None:
            progress.advance(task)
        return result

    with (
        temporary_task(progress, "Downloading items", total=len(items))
        if progress is not None
        else nullcontext(None)
    ) as task:
        return list(await asyncio.gather(*(_download_one(item, task) for item in items)))
