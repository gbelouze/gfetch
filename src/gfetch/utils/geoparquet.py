"""GeoParquet output helpers."""

import os
import tempfile
from pathlib import Path

import geopandas as gpd

__all__ = ["write_geoparquet"]


def write_geoparquet(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """
    Write a GeoDataFrame to GeoParquet atomically.

    Written under a hidden temporary name in the same directory, then renamed into
    place, so `path` existing means the write completed.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Data to write.
    path : Path
        Destination file, overwritten if present.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        gdf.to_parquet(tmp)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
