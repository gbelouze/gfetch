"""Download stage: fetch STAC item assets to a local cache, atomically and resumably.

Downloads land at ``<cache_dir>/<item_id>/<asset_key><suffix>``. Each completed asset
is marked by a sibling ``<cache_dir>/<item_id>/<asset_key>.complete`` sentinel file, so
a killed/preempted job can resume by skipping assets whose sentinel already exists,
without needing a central manifest or any locking, so many workers can share one cache
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
from typing import Any
from urllib.parse import urlsplit

import aiohttp
import pystac
import stac_asset.http_client
from stac_asset import Config, DownloadError, download_item
from stac_asset.messages import (
    ErrorAssetDownload,
    FinishAssetDownload,
    Message,
    OpenUrl,
    SkipAssetDownload,
    WriteChunk,
)

from gfetch.utils.progress import Progress, TaskID, temporary_task

log = logging.getLogger(__name__)


def _patch_stac_asset_proxy_support() -> None:
    """
    Make stac-asset's downloads respect `HTTP_PROXY`/`HTTPS_PROXY`.

    `stac_asset.HttpClient` builds its `aiohttp.ClientSession` with `trust_env`'s
    default (`False`). Unlike `requests`/`curl`, `aiohttp` then silently ignores
    proxy environment variables and attempts a direct connection, which just hangs
    forever on a network that requires a proxy for egress (e.g. many HPC compute
    nodes) rather than raising an error. `stac_asset` exposes no config option for
    this, so this replaces the `ClientSession` constructor it calls with one that
    defaults `trust_env=True`.
    """

    def _client_session(*args: Any, trust_env: bool = True, **kwargs: Any) -> aiohttp.ClientSession:
        # A plain factory, not a ClientSession subclass: aiohttp explicitly
        # discourages subclassing it, and stac_asset only ever calls this as a
        # constructor-shaped callable, never checks its type.
        return aiohttp.ClientSession(*args, trust_env=trust_env, **kwargs)

    # Deliberately swapping a class for a constructor-shaped factory function,
    # not expressible as a type[ClientSession] since it isn't one.
    stac_asset.http_client.ClientSession = _client_session  # type: ignore[assignment]


_patch_stac_asset_proxy_support()

__all__ = ["download_items"]

_RETRY_TRIES = 5
_RETRY_INITIAL_DELAY = 1.0
_RETRY_BACKOFF = 1.7


def _s3_uri_to_public_https(href: str) -> str:
    """
    Rewrite an `s3://bucket/key` href to its public `https://bucket.s3.amazonaws.com/key`
    equivalent.

    `stac_asset` routes `s3://` hrefs through its `S3Client`, which hardcodes its
    default region to 'us-west-2' (`stac_asset.config.DEFAULT_S3_REGION_NAME`)
    regardless of the bucket's actual region: every request to a bucket hosted
    elsewhere (e.g. Sentinel-1's `sentinel-s1-l1c`, actually in `eu-central-1`) pays a
    wrong-region redirect round trip on every single request (confirmed: ~3x slower
    end to end for that bucket). The public virtual-hosted-style URL is served from
    the global S3 endpoint instead, with no region to get wrong, and is routed through
    `stac_asset`'s plain `HttpClient`, which sidesteps the problem for any public
    bucket, regardless of its actual region.

    Parameters
    ----------
    href : str
        An asset href.

    Returns
    -------
    str
        `href` unchanged if it isn't an `s3://` URI, otherwise its public HTTPS
        equivalent.
    """
    parsed = urlsplit(href)
    if parsed.scheme != "s3":
        return href
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def _strip_query(url: str) -> str:
    """
    Drop the query string and fragment of a URL (e.g. a Planetary Computer SAS token).

    Parameters
    ----------
    url : str
        URL to strip.

    Returns
    -------
    str
        `url` without its query string and fragment.
    """
    return urlsplit(url)._replace(query="", fragment="").geturl()


def _missing_asset_keys(error: DownloadError, item: pystac.Item) -> set[str] | None:
    """
    Find which of `item`'s assets a download failed on because they don't exist.

    Planetary Computer occasionally lists assets whose blob is missing from its
    storage account, which consistently answers 404 for them.

    Parameters
    ----------
    error : DownloadError
        Error raised by `download_item()` for `item`.
    item : pystac.Item
        Item whose download raised `error`.

    Returns
    -------
    set[str] | None
        Keys of the assets that 404'd, or None if `error` holds any other kind of
        failure, or a 404 that can't be traced back to one of `item`'s assets.
    """
    keys_by_url = {
        _strip_query(_s3_uri_to_public_https(asset.href)): key for key, asset in item.assets.items()
    }
    missing = set()
    for e in error.exceptions:
        if not (isinstance(e, aiohttp.ClientResponseError) and e.status == 404):
            return None
        key = keys_by_url.get(_strip_query(str(e.request_info.real_url)))
        if key is None:
            return None
        missing.add(key)
    return missing


def _rewrite_s3_hrefs(item: pystac.Item) -> pystac.Item:
    """
    Deep-copy an item with every `s3://` asset href rewritten to its public HTTPS
    equivalent (see `_s3_uri_to_public_https`).

    Parameters
    ----------
    item : pystac.Item
        Item whose asset hrefs may include `s3://` URIs.

    Returns
    -------
    pystac.Item
        A deep copy of `item`, safe to pass to `download_item()` (which mutates its
        input), with `s3://` hrefs rewritten.
    """
    item = copy.deepcopy(item)
    for asset in item.assets.values():
        asset.href = _s3_uri_to_public_https(asset.href)
    return item


async def _retry_async[T](
    coro_fn: Callable[[], Awaitable[T]],
    *,
    tries: int = _RETRY_TRIES,
    delay: float = _RETRY_INITIAL_DELAY,
    backoff: float = _RETRY_BACKOFF,
    give_up: Callable[[Exception], bool] = lambda _: False,
) -> T:
    """
    Retry an async callable with exponential backoff.

    The synchronous `retry` package used elsewhere in gfetch's sibling projects only
    wraps regular callables: it never awaits a coroutine, so it can't retry
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
    give_up : Callable[[Exception], bool]
        Predicate on a raised exception, true if it is permanent and must be re-raised
        right away. Defaults to never giving up early.

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
            if attempt >= tries or give_up(e):
                raise
            log.warning(f"Attempt {attempt}/{tries} failed ({e}), retrying in {delay:.1f}s")
            await asyncio.sleep(delay)
            delay *= backoff


async def _report_asset_progress(
    messages: asyncio.Queue[Message], progress: Progress, task: TaskID, n_assets: int
) -> None:
    """
    Drive one item's progress bar from its `stac_asset.download_item()` message stream.

    Runs concurrently with the `download_item()` call it was handed the other end of
    `messages` for, terminating once every asset in that call has reported a terminal
    message. `download_item()` itself never signals "queue done", so this counts
    terminal messages instead of waiting for one.

    Parameters
    ----------
    messages : asyncio.Queue[Message]
        The `messages` queue passed to `download_item()`.
    progress : Progress
        Progress tracker owning `task`.
    task : TaskID
        This item's own bar, already added by the caller.
    n_assets : int
        Number of assets being downloaded in this call: how many terminal messages
        (finish/error/skip) to wait for before returning.
    """
    total = 0
    done = 0
    while done < n_assets:
        message = await messages.get()
        if isinstance(message, OpenUrl) and message.size:
            total += message.size
            progress.update(task, total=total)
        elif isinstance(message, WriteChunk):
            progress.advance(task, message.size)
        elif isinstance(message, FinishAssetDownload | ErrorAssetDownload | SkipAssetDownload):
            done += 1


async def _download_item(
    item: pystac.Item,
    cache_dir: Path,
    asset_keys: Sequence[str],
    config: Config,
    semaphore: asyncio.Semaphore,
    progress: Progress | None = None,
) -> pystac.Item | None:
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
    progress : Progress | None
        Rich progress tracker. When given, this item gets its own byte-progress bar
        for the duration of its download (added/removed here, self-contained, with no
        separate progress-reporting task elsewhere). Defaults to None.

    Returns
    -------
    pystac.Item | None
        A copy of `item` whose requested asset hrefs point at the local cache instead
        of the remote source, ready to be handed to `odc.stac.load()`. Assets missing
        from the remote storage (404) are logged and left out, and None is returned if
        that leaves no asset at all.
    """
    item_dir = cache_dir / item.id
    item_dir.mkdir(parents=True, exist_ok=True)

    available_keys = [k for k in asset_keys if k in item.assets]
    pending_keys = [k for k in available_keys if not (item_dir / f"{k}.complete").exists()]

    missing_keys: set[str] = set()

    async def _attempt(keys: list[str]) -> pystac.Item:
        # The move out of the temp dir must happen before it's cleaned up on exiting
        # this `with` block, so it stays inside `_attempt` (retried as a whole) rather
        # than after it. A fresh deep copy of `item` is required per attempt too:
        # download_item() mutates its input item in place (sets its self href,
        # rewrites asset hrefs), so retrying against the same object would hand a
        # failed attempt's corrupted state to the next one.
        log.debug(f"{item.id}: starting download of {keys}")
        download_config = dataclasses.replace(config, include=keys)
        with tempfile.TemporaryDirectory(dir=item_dir) as tmp:
            if progress is None:
                downloaded = await download_item(
                    _rewrite_s3_hrefs(item),
                    Path(tmp),
                    config=download_config,
                    keep_non_downloaded=True,
                )
            else:
                messages: asyncio.Queue[Message] = asyncio.Queue()
                with temporary_task(progress, item.id, total=None, bytes=True) as task:
                    downloaded, _ = await asyncio.gather(
                        download_item(
                            _rewrite_s3_hrefs(item),
                            Path(tmp),
                            config=download_config,
                            keep_non_downloaded=True,
                            messages=messages,
                        ),
                        _report_asset_progress(messages, progress, task, len(keys)),
                    )
            for key in keys:
                # .get_absolute_href() (not .href) is required here: download_item()
                # always relative-ifies asset hrefs once it sets a self href, which it
                # always does when writing into a directory.
                href = downloaded.assets[key].get_absolute_href()
                assert href is not None, f"downloaded asset {key} has no absolute href"
                tmp_path = Path(href)
                final_path = item_dir / f"{key}{tmp_path.suffix}"
                shutil.move(tmp_path, final_path)
                (item_dir / f"{key}.complete").touch()
                downloaded.assets[key].href = str(final_path)
                log.debug(f"Downloaded {item.id}/{key} -> {final_path}")
        return downloaded

    def _is_missing_asset(e: Exception) -> bool:
        return isinstance(e, DownloadError) and _missing_asset_keys(e, item) is not None

    result = None
    if pending_keys:
        async with semaphore:
            while keys := [k for k in pending_keys if k not in missing_keys]:
                try:
                    result = await _retry_async(lambda: _attempt(keys), give_up=_is_missing_asset)
                    break
                except DownloadError as e:
                    newly_missing = _missing_asset_keys(e, item)
                    if newly_missing is None:
                        raise
                    log.warning(
                        f"{item.id}: skipping asset(s) {sorted(newly_missing)}, missing "
                        f"from the remote storage (404): {e}"
                    )
                    missing_keys |= newly_missing
    else:
        log.debug(f"{item.id}: all requested assets already cached, skipping download")
    if result is None:
        result = copy.deepcopy(item)

    available_keys = [k for k in available_keys if k not in missing_keys]
    if missing_keys and not available_keys:
        log.warning(f"{item.id}: no requested asset left to download, dropping item")
        return None
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
        Rich progress tracker: one main bar advanced per completed item, plus one
        byte-progress bar per item currently downloading (up to `max_concurrent_items`
        at once), all on this same tracker, since rich requires a single `Progress`
        instance to render multiple concurrent bars correctly. Defaults to None (no
        progress bars).

    Returns
    -------
    list[pystac.Item]
        Copies of `items` whose requested asset hrefs point at the local cache, in
        the same order as `items`, ready to be handed to `odc.stac.load()`. Assets
        missing from the remote storage (404) are logged and left out, and so are
        items left with no asset at all.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = config or Config()
    semaphore = asyncio.Semaphore(max_concurrent_items)
    log.info(
        f"Downloading {len(items)} item(s) x {len(asset_keys)} asset(s) into {cache_dir} "
        f"(max_concurrent_items={max_concurrent_items})"
    )

    n_done = 0

    async def _download_one(item: pystac.Item, task: TaskID | None) -> pystac.Item | None:
        nonlocal n_done
        result = await _download_item(item, cache_dir, asset_keys, config, semaphore, progress)
        n_done += 1
        status = "cached" if result is not None else "dropped"
        log.info(f"{item.id}: {status} [{n_done}/{len(items)}]")
        if progress is not None and task is not None:
            progress.advance(task)
        return result

    with (
        temporary_task(progress, "Downloading items", total=len(items))
        if progress is not None
        else nullcontext(None)
    ) as task:
        results = await asyncio.gather(*(_download_one(item, task) for item in items))
    result = [r for r in results if r is not None]
    log.info(f"Downloaded {len(result)} item(s) into {cache_dir}")
    return result
