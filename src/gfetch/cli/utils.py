import logging
from pathlib import Path

from gfetch.mosaic import resolve_chunks, resolve_shards
from gfetch.rechunk import rechunk as rechunk_store
from gfetch.utils.progress import count_bar
from gfetch.utils.system import available_cpus

log = logging.getLogger(__name__)


def _layout(chunk: int | None, shard_factor: int | None) -> tuple[int, int]:
    """Chunk and shard side in pixels, `mosaic`'s defaults filling in for None."""
    chunks = resolve_chunks({"x": chunk, "y": chunk} if chunk is not None else None)
    shards = resolve_shards(shard_factor, chunks)
    return chunks["x"], shards["x"] if shards is not None else chunks["x"]


def rechunk(
    stores: list[Path],
    *,
    chunk: int | None = None,
    shard_factor: int | None = None,
    bands: list[str] | None = None,
    output: Path | None = None,
    task_id: int = 0,
    n_tasks: int = 1,
    n_workers: int | None = None,
) -> None:
    """
    Rewrite Zarr stores in `mosaic`'s stacked layout and a new chunk and shard size,
    one after the other.

    Parameters
    ----------
    stores : list[Path]
        Zarr stores to rechunk.
    chunk : int | None
        Chunk side along `y` and `x`, in pixels. Defaults to None, which uses
        `gfetch.mosaic.resolve_chunks`' default.
    shard_factor : int | None
        Number of chunks per shard along `y` and `x`. Defaults to None, which uses
        `gfetch.mosaic.resolve_shards`' default.
    bands : list[str] | None
        Band order of the new stores, see `gfetch.rechunk.rechunk`. Defaults to None.
    output : Path | None
        Where to write the new store, only with a single store. Defaults to None,
        which replaces each store in place.
    task_id : int
        This invocation's index among `n_tasks` concurrent invocations. Defaults to 0.
    n_tasks : int
        Total number of concurrent invocations. Defaults to 1.
    n_workers : int | None
        Number of copying threads. Defaults to None, which uses
        `gfetch.utils.system.available_cpus`.

    Raises
    ------
    ValueError
        If `output` is given along with several stores.
    """
    if output is not None and len(stores) != 1:
        raise ValueError(f"--output needs exactly one store, got {len(stores)}.")
    chunk_size, shard = _layout(chunk, shard_factor)
    workers = n_workers if n_workers is not None else available_cpus()
    log.info(f"Target layout: {chunk_size} px chunks, {shard} px shards, {workers} thread(s)")
    with count_bar() as progress:
        for i, store in enumerate(stores, start=1):
            log.info(f"Rechunking {store} [{i}/{len(stores)}]")
            rechunk_store(
                store,
                chunk_size,
                shard,
                bands=bands,
                output=output,
                task_id=task_id,
                n_tasks=n_tasks,
                n_workers=workers,
                progress=progress,
            )


def check(
    stores: list[Path],
    *,
    chunk: int | None = None,
    shard_factor: int | None = None,
    bands: list[str] | None = None,
) -> bool:
    """
    Check that Zarr stores have the form `gfetch mosaic` writes, logging each check.

    Parameters
    ----------
    stores : list[Path]
        Zarr stores to check.
    chunk : int | None
        Expected chunk side along `y` and `x`, in pixels. Defaults to None, which uses
        `gfetch.mosaic.resolve_chunks`' default.
    shard_factor : int | None
        Expected number of chunks per shard along `y` and `x`. Defaults to None, which
        uses `gfetch.mosaic.resolve_shards`' default.
    bands : list[str] | None
        Expected band names, in order. Defaults to None, which accepts any.

    Returns
    -------
    bool
        True if every store passed every check.
    """
    from gfetch.check import check_store

    chunk_size, shard = _layout(chunk, shard_factor)
    n_ok = 0
    for i, store in enumerate(stores, start=1):
        results = check_store(store, chunk_size, shard, bands)
        ok = all(r.ok for r in results)
        n_ok += ok
        log.info(f"{store}: {'OK' if ok else 'FAILED'} [{i}/{len(stores)}]")
        for r in results:
            line = f"  {'✓' if r.ok else '✗'} {r.name}" + (f": {r.detail}" if r.detail else "")
            if r.ok:
                log.info(line)
            else:
                log.error(line)
    log.info(f"{n_ok}/{len(stores)} store(s) passed")
    return n_ok == len(stores)
