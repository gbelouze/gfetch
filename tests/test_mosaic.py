import numpy as np
import pytest
from odc.geo.geobox import GeoBox

from gfetch.mosaic import mosaic
from gfetch.profiles import get_profile
from gfetch.search import search
from gfetch.sources import get_source


@pytest.mark.slow
def test_mosaic_against_earthsearch() -> None:
    bbox = (2.30, 48.85, 2.33, 48.87)  # small AOI, keeps the test fast
    source = get_source("earthsearch")
    profile = get_profile("sentinel-2")

    items = search(
        source,
        "sentinel-2",
        bbox=bbox,
        datetime="2026-06-01/2026-06-30",
        query={"eo:cloud_cover": {"lt": 40}},
    )
    assert items

    geobox = GeoBox.from_bbox(bbox, crs="utm", resolution=60.0)  # coarse, keeps it fast
    ds = mosaic(
        items,
        geobox,
        ["red"],
        mask_band=profile.cloud_mask_band,
        mask_out=profile.cloud_mask_out,
    )
    computed = ds.compute()

    assert "red" in computed.data_vars
    assert "time" not in computed.dims  # composited away
    assert np.isfinite(computed["red"].values).any()
