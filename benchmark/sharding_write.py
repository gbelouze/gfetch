"""Benchmark `mosaic`'s write strategies: unsharded, sharded with every band in one
write, and sharded with one band per write (what `gfetch <satellite> mosaic` does).

Synthetic workload shaped like the real one, without any network or disk input: per
band, a uint16 `(time, y, x)` stack, masked by a classification layer shared by all
bands (like Sentinel-2's SCL), reduced by a median over time. Writes go through
`gfetch.write`'s real `prepare_template`/`write_regions`/`write_region`, one unit of
work at a time, like the `mosaic` command's loop.

Each strategy runs in its own subprocess, so its peak RSS isn't inflated by the
strategies run before it. Reports wall time of the writes (template excluded), peak
RSS, peak RSS above the process's baseline after imports, and the number of files
written.

Depends on the `gfetch` package itself, like `gfetch_pipeline.py`.
"""

# %%
import csv
import json
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal

import cyclopts
import dask
import dask.array as da
import numpy as np
import xarray as xr

from gfetch.write import prepare_template, write_region, write_regions

RESULTS_DIR = Path(__file__).parent / "results"

Strategy = Literal["unsharded", "sharded", "sharded-per-band"]
STRATEGIES: tuple[Strategy, ...] = ("unsharded", "sharded", "sharded-per-band")

app = cyclopts.App()


def _peak_rss_mb() -> float:
    """
    Peak resident set size of this process so far.

    Returns
    -------
    float
        Peak RSS in MB (`ru_maxrss` is bytes on macOS, KB on Linux).
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def _build(region: dict[str, slice], bands: list[str], n_time: int, chunk: int) -> xr.Dataset:
    """
    Lazily build a synthetic masked median composite over one region.

    Parameters
    ----------
    region : dict[str, slice]
        `{"y": slice, "x": slice}` region of the full array, with explicit bounds.
    bands : list[str]
        Band names to build; each gets its own random stack.
    n_time : int
        Number of time steps per stack.
    chunk : int
        Spatial dask chunk size.

    Returns
    -------
    xr.Dataset
        Dask-backed `(y, x)` float32 composite per band, chunked at `chunk`.
    """
    ny = region["y"].stop - region["y"].start
    nx = region["x"].stop - region["x"].start
    chunks = (n_time, chunk, chunk)
    cloudy = da.random.RandomState(0).random_sample((n_time, ny, nx), chunks=chunks) < 0.3
    data = {}
    for i, band in enumerate(bands):
        stack = da.random.RandomState(i + 1).randint(1, 10_000, (n_time, ny, nx), chunks=chunks)
        masked = xr.DataArray(stack.astype("uint16"), dims=("time", "y", "x")).where(
            ~xr.DataArray(cloudy, dims=("time", "y", "x"))
        )
        data[band] = masked.median("time").astype("float32")
    coords = {
        "y": np.arange(region["y"].start, region["y"].stop),
        "x": np.arange(region["x"].start, region["x"].stop),
    }
    return xr.Dataset(data, coords=coords).chunk({"y": chunk, "x": chunk})


@app.command
def run_one(
    strategy: Strategy,
    size: int,
    chunk: int,
    shard_factor: int,
    n_bands: int,
    n_time: int,
    n_threads: int,
) -> None:
    """
    Run one strategy in this process and print its measurements as JSON.

    Parameters
    ----------
    strategy : Strategy
        Write strategy to run.
    size : int
        Output array side, in pixels.
    chunk : int
        Chunk side, in pixels.
    shard_factor : int
        Chunks per shard along each side (ignored by `unsharded`).
    n_bands : int
        Number of bands.
    n_time : int
        Time steps per band stack.
    n_threads : int
        Dask threads.
    """
    baseline_mb = _peak_rss_mb()
    bands = [f"b{i}" for i in range(n_bands)]
    shards = (
        None if strategy == "unsharded" else {"y": shard_factor * chunk, "x": shard_factor * chunk}
    )
    full = {"y": slice(0, size), "x": slice(0, size)}
    tmp = Path(tempfile.mkdtemp())
    path = tmp / "store.zarr"
    try:
        with dask.config.set(scheduler="threads", num_workers=n_threads):
            prepare_template(_build(full, bands, n_time, chunk), path, shards=shards)
            units = write_regions(path, bands[0])
            start = time.perf_counter()
            for region in units:
                bounded = {dim: slice(sl.start, sl.stop) for dim, sl in region.items()}
                if strategy == "sharded-per-band":
                    for band in bands:
                        write_region(_build(bounded, [band], n_time, chunk), path, region)
                else:
                    write_region(_build(bounded, bands, n_time, chunk), path, region)
            elapsed = time.perf_counter() - start
        n_files = sum(1 for p in path.rglob("c/**/*") if p.is_file())
    finally:
        shutil.rmtree(tmp)
    peak_mb = _peak_rss_mb()
    print(
        json.dumps(
            {
                "strategy": strategy,
                "n_units": len(units),
                "write_seconds": round(elapsed, 2),
                "peak_rss_mb": round(peak_mb),
                "peak_above_baseline_mb": round(peak_mb - baseline_mb),
                "n_files": n_files,
            }
        )
    )


@app.default
def main(
    size: int = 4096,
    chunk: int = 256,
    shard_factor: int = 16,
    n_bands: int = 6,
    n_time: int = 12,
    n_threads: int = 4,
    output: Path = RESULTS_DIR / "sharding_write.csv",
) -> None:
    """
    Run every strategy, each in its own subprocess, and write a CSV of the results.

    Parameters
    ----------
    size : int
        Output array side, in pixels. Defaults to 4096.
    chunk : int
        Chunk side, in pixels. Defaults to 256.
    shard_factor : int
        Chunks per shard along each side. Defaults to 16.
    n_bands : int
        Number of bands. Defaults to 6.
    n_time : int
        Time steps per band stack. Defaults to 12.
    n_threads : int
        Dask threads. Defaults to 4.
    output : Path
        Results CSV path. Defaults to `results/sharding_write.csv`.
    """
    params = {
        "size": size,
        "chunk": chunk,
        "shard_factor": shard_factor,
        "n_bands": n_bands,
        "n_time": n_time,
        "n_threads": n_threads,
    }
    shard_mb = (shard_factor * chunk) ** 2 * 4 / 1e6
    print(
        f"{params} -> one band's shard is {shard_mb:.0f} MB as float32, "
        f"all bands {shard_mb * n_bands:.0f} MB"
    )
    rows = []
    for strategy in STRATEGIES:
        args = [f"--{k.replace('_', '-')}={v}" for k, v in params.items()]
        result = subprocess.run(
            [sys.executable, __file__, "run-one", strategy, *args],
            capture_output=True,
            text=True,
            check=True,
        )
        row = {**params, **json.loads(result.stdout.strip().splitlines()[-1])}
        print(row)
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    app()
