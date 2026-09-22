"""Write stage: persist a mosaic to a local Zarr store.

Default output target is plain Zarr (not Icechunk) with pre-planned, non-overlapping
per-worker chunk regions - safe on any POSIX filesystem, including shared HPC storage.
See `claude/tech-stack.md`'s "HPC / distributed execution" section for why.
"""

import logging
from pathlib import Path
from typing import Literal

import xarray as xr

from gfetch.utils.memory import log_chunk_footprint

log = logging.getLogger(__name__)

__all__ = ["prepare_template", "write", "write_region"]

_ZarrMode = Literal["a", "a-", "r", "r+", "w", "w-"]


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
    log.debug(f"Writing dataset {dict(ds.sizes)} to {path} (mode={mode})")
    log_chunk_footprint(ds, log)
    ds.to_zarr(path, mode=mode)
    log.info(f"Wrote dataset to {path}")


def prepare_template(ds: xr.Dataset, path: Path) -> None:
    """
    Write only a Zarr store's metadata and coordinates, without any chunk data.

    Coordinating step for a pre-planned disjoint-region write: run this once before
    any worker calls `write_region`, so the store's shape/coords/chunking exist before
    workers start writing their own non-overlapping regions.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset whose shape/coords/chunking define the store; its data isn't written.
    path : Path
        Destination Zarr store path.
    """
    ds.to_zarr(path, compute=False, mode="w")
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
        e.g. `{"x": slice(0, 512), "y": slice(None)}`.
    """
    # xarray's region-write rejects any variable lacking a dimension in common with
    # `region` - scalar coordinates like a CRS grid mapping variable must be dropped,
    # since they were already written once by prepare_template.
    ds = ds.drop_vars([c for c in ds.coords if c not in ds.dims])
    log_chunk_footprint(ds, log)
    ds.to_zarr(path, region=region)
    log.debug(f"Wrote region {region} to {path}")
