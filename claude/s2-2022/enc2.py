import warnings

warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pystac_client
import rasterio

# Every `sentinel-2-l2a` item from baseline 04.00 on with `boa_offset_applied: false`, over
# the four `enc.py` boxes: which of them carry the +1000 offset in their DNs.
c = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
aois = {
    "FR": [1.5, 46.5, 4.5, 48.5],
    "MZ": [34.0, -19.5, 35.0, -18.5],
    "JP": [138.5, 35.0, 140.5, 36.5],
    "US": [-91.0, 38.0, -89.0, 39.5],
}
items = []
for aoi, bbox in aois.items():
    for i in c.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime="2022-01-25/2026-10-15",
        query={"earthsearch:boa_offset_applied": {"eq": False}},
    ).items():
        items.append((aoi, i))
print(len(items), "items")


# Only SCL vegetation (4) and water (6) pixels: clear and dark in blue, so their DNs sit
# below 1000 without the offset and above it with. Clouds would read as +1000 either way.
def stat(entry):
    aoi, i = entry
    try:
        with rasterio.open(i.assets["blue"].href) as src:
            a = src.read(1, out_shape=(549, 549)).astype(float)
        with rasterio.open(i.assets["scl"].href) as src:
            scl = src.read(1, out_shape=(549, 549))
    except rasterio.errors.RasterioError:
        return aoi, i, None
    v = a[np.isin(scl, [4, 6]) & (a > 0)]
    if v.size < 2000:
        return aoi, i, None
    return aoi, i, (v < 1000).mean()


with ThreadPoolExecutor(8) as ex:
    rows = list(ex.map(stat, items))
for aoi, i, f in sorted(rows, key=lambda r: r[1].datetime):
    enc = "?" if f is None else "+1000" if f < 0.02 else "none" if f > 0.5 else "?"
    print(
        f"{aoi} {i.datetime:%Y-%m-%d} {i.id:32} bl={i.properties['s2:processing_baseline']} "
        f"cc={i.properties['eo:cloud_cover']:5.1f} created={i.properties.get('created', '')[:10]} "
        f"frac<1000={'-' if f is None else f'{f:.3f}'} {enc}"
    )
