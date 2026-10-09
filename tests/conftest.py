from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from gfetch.write import prepare_template, stack_bands, write_region, write_regions

BANDS = ["red", "green", "blue"]
Skip = Callable[[dict[str, slice]], bool]


class Stores:
    """Builds small Zarr stores the way `mosaic` writes them, current or older layouts."""

    @staticmethod
    def per_band() -> xr.Dataset:
        """40x36 px bands, the top-left 8x8 px all NaN, as over the sea."""
        geobox = GeoBox.from_bbox(
            (500_000, 9_000_000, 500_360, 9_000_400), "EPSG:32736", resolution=10
        )
        rng = np.random.default_rng(0)
        data = {}
        for band in BANDS:
            values = rng.random(geobox.shape).astype("float32")
            values[:8, :8] = np.nan
            data[band] = (("y", "x"), values)
        return xr.Dataset(data, coords=xr_coords(geobox), attrs={"source": "test"})

    @staticmethod
    def write(ds: xr.Dataset, path: Path, chunk: int, shard: int, skip: Skip | None = None) -> None:
        """Write `ds` the way `mosaic` does, one shard at a time."""
        prepare_template(
            ds.chunk({"y": shard, "x": shard}),
            path,
            shards={"y": shard, "x": shard} if shard != chunk else None,
            chunks={"y": chunk, "x": chunk},
            skip=skip if skip is not None else lambda region: False,
        )
        var = next(iter(ds.data_vars))
        for region in write_regions(path, str(var)):
            write_region(ds.isel(region), path, region)

    def old(self, path: Path, skip: Skip | None = None) -> None:
        """A per-band store with 4 px chunks in 16 px shards."""
        self.write(self.per_band(), path, 4, 16, skip)

    def stacked(
        self,
        path: Path,
        bands: Sequence[str] = tuple(BANDS),
        skip: Skip | None = None,
        chunk: int = 4,
        shard: int = 16,
    ) -> None:
        """A stacked store, by default with 4 px chunks in 16 px shards."""
        self.write(stack_bands(self.per_band(), bands), path, chunk, shard, skip)


@pytest.fixture
def stores() -> Stores:
    return Stores()
