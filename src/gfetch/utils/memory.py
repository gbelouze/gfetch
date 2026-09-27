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
    shapes: list[tuple[int, ...]] = []
    chunk_shapes: list[tuple[int, ...]] = []
    footprints: dict[str, str] = {}
    for name, var in ds.data_vars.items():
        data = var.data
        shapes.append(var.shape)
        footprints[str(name)] = f"{var.nbytes / 1e9:.2f} GB"
        if hasattr(data, "chunksize"):
            chunk_shapes.append(data.chunksize)
            total_chunk_bytes += math.prod(data.chunksize) * data.dtype.itemsize
    log.debug(f"shape {_largest(shapes)}, chunks {_largest(chunk_shapes)}, footprint {footprints}")
    log.info(
        f"Estimated peak memory ({n_workers} concurrent chunk(s) across all "
        f"variables): {total_chunk_bytes * n_workers / 1e9:.2f} GB"
    )


def _largest(shapes: list[tuple[int, ...]]) -> str:
    """
    Describe the largest of several shapes and how many share it.

    Parameters
    ----------
    shapes : list[tuple[int, ...]]
        One shape per variable.

    Returns
    -------
    str
        E.g. `(23, 256, 256) x 12/13`, or `none` if `shapes` is empty.
    """
    if not shapes:
        return "none"
    largest = max(shapes, key=math.prod)
    return f"{largest} x {shapes.count(largest)}/{len(shapes)}"
