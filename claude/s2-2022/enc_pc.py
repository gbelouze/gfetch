import warnings

warnings.filterwarnings("ignore")
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pystac_client
import rasterio

from gfetch.sources import planetary_computer_signer

# `enc2.py`'s per-item DN-encoding test, on Planetary Computer's `sentinel-2-l2a`: a few
# items per (baseline, month) over two small boxes, from 2017 to 2026.
c = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
sign = planetary_computer_signer()
aois = {"MZ": [34.3, -19.2, 34.7, -18.8], "US": [-90.2, 38.6, -89.8, 39.0]}
months = [
    "2017-07",
    "2018-07",
    "2019-07",
    "2020-07",
    "2021-07",
    "2022-01",
    "2022-02",
    "2022-03",
    "2022-07",
    "2023-07",
    "2024-07",
    "2025-07",
    "2026-07",
]
items = []
for aoi, bbox in aois.items():
    for m in months:
        found = list(
            c.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=f"{m}-01/{m}-28",
                query={"eo:cloud_cover": {"lt": 40}},
                max_items=4,
            ).items()
        )
        items += [(aoi, m, i) for i in found]
print(len(items), "items")


def stat(entry):
    aoi, m, i = entry
    try:
        with rasterio.open(sign(i.assets["B02"].href)) as src:
            a = src.read(1, out_shape=(549, 549)).astype(float)
        with rasterio.open(sign(i.assets["SCL"].href)) as src:
            scl = src.read(1, out_shape=(549, 549))
    except rasterio.errors.RasterioError:
        return entry, None
    v = a[np.isin(scl, [4, 6]) & (a > 0)]
    return entry, (v < 1000).mean() if v.size >= 2000 else None


with ThreadPoolExecutor(8) as ex:
    rows = list(ex.map(stat, items))
groups = defaultdict(list)
for (aoi, m, i), f in rows:
    enc = "?" if f is None else "+1000" if f < 0.02 else "none" if f > 0.5 else "?"
    rb = i.assets["B02"].extra_fields.get("raster:bands", [{}])[0]
    print(
        f"{aoi} {i.datetime:%Y-%m-%d} {i.id:45} bl={i.properties.get('s2:processing_baseline')} "
        f"offset={rb.get('offset')} frac<1000={'-' if f is None else f'{f:.3f}'} {enc}"
    )
    groups[(i.properties.get("s2:processing_baseline"), rb.get("offset"))].append(enc)
for k in sorted(groups, key=str):
    print(k, {e: groups[k].count(e) for e in set(groups[k])})
