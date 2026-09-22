"""Diagnostics for anticipating dask compute/write peak memory usage."""

import logging
import math

import dask
import xarray as xr

from gfetch.utils.system import available_cpus

__all__ = ["log_chunk_footprint"]


def log_chunk_footprint(ds: xr.Dataset, log: logging.Logger) -> None:
    """
    Log each data variable's chunk shape and an estimated worst-case peak memory
    footprint, at INFO level.

    The peak-memory estimate is `sum(per-variable chunk bytes) * dask worker count` -
    the case where every worker is simultaneously materializing one chunk of every
    variable. Two things it does not capture: decode/decompression overhead in the
    underlying rasterio/GDAL reads, and the extra memory a reduction that isn't
    chunk-wise associative (e.g. `median`, `gfetch.mosaic.composite`'s default) needs
    to gather its whole reduction axis into memory per spatial chunk. Call this on
    `ds` *before* such a reduction to see that cost - calling it only on the already-
    reduced result (as it has no dimension left to gather) understates the real peak.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to inspect, still lazy (dask-backed).
    log : logging.Logger
        Logger to write to, namespaced to the caller's module.
    """
    n_workers = dask.config.get("num_workers", None) or available_cpus()
    log.info(f"Dask worker threads: {n_workers} (available CPUs={available_cpus()})")

    total_chunk_bytes = 0
    for name, var in ds.data_vars.items():
        data = var.data
        if not hasattr(data, "chunksize"):
            log.debug(f"{name}: not dask-backed (already in memory), shape={var.shape}")
            continue
        chunk_bytes = math.prod(data.chunksize) * data.dtype.itemsize
        total_chunk_bytes += chunk_bytes
        log.debug(
            f"{name}: dtype={data.dtype}, shape={var.shape}, chunks={data.chunksize} "
            f"({data.npartitions} chunk(s) total, {chunk_bytes / 1e6:.1f} MB/chunk), "
            f"total={data.nbytes / 1e9:.2f} GB"
        )
    log.info(
        f"Estimated peak memory ({n_workers} concurrent chunk(s) across all "
        f"variables): {total_chunk_bytes * n_workers / 1e9:.2f} GB"
    )
