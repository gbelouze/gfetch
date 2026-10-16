import warnings

warnings.filterwarnings("ignore")
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pystac_client
import rasterio

# Classifies each item's DN encoding from its pixels: blue reflectance over dark targets
# is ~0.01-0.05, so without the +1000 offset a sizeable share of valid blue DNs sit below
# 1000, and with it almost none do. Reads one coarse overview per item.
c = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
aois = {
    "FR": [1.5, 46.5, 4.5, 48.5],
    "MZ": [34.0, -19.5, 35.0, -18.5],
    "JP": [138.5, 35.0, 140.5, 36.5],
    "US": [-91.0, 38.0, -89.0, 39.5],
}
months = [
    "2017-03",
    "2017-09",
    "2018-06",
    "2019-01",
    "2019-07",
    "2020-07",
    "2021-07",
    "2022-01",
    "2022-03",
    "2022-07",
    "2023-03",
    "2023-09",
    "2024-07",
    "2025-07",
    "2026-07",
]
groups = defaultdict(list)
for aoi, bbox in aois.items():
    for m in months:
        for coll in ("sentinel-2-l2a", "sentinel-2-c1-l2a"):
            for i in c.search(
                collections=[coll],
                bbox=bbox,
                datetime=f"{m}-01/{m}-28",
                query={"eo:cloud_cover": {"lt": 30}},
                max_items=200,
            ).items():
                key = (
                    coll,
                    i.properties.get("s2:processing_baseline"),
                    i.properties.get("earthsearch:boa_offset_applied"),
                    i.assets["blue"].extra_fields.get("raster:bands", [{}])[0].get("offset"),
                )
                groups[key].append((aoi, i))


def stat(entry):
    aoi, i = entry
    with rasterio.open(i.assets["blue"].href, OVERVIEW_LEVEL=2) as src:
        a = src.read(1).astype(float)
    v = a[a > 0]
    if v.size < 1000:
        return None
    return aoi, i.id, np.percentile(v, 1), (v < 1000).mean()


random.seed(0)
for key in sorted(groups, key=str):
    sample = random.sample(groups[key], min(8, len(groups[key])))
    with ThreadPoolExecutor(8) as ex:
        rows = [r for r in ex.map(stat, sample) if r]
    verdict = sorted(
        {"+1000" if f < 0.01 and p1 > 900 else "none" if f > 0.05 else "?" for _, _, p1, f in rows}
    )
    print(
        f"{key[0]:18} bl={key[1]} applied={key[2]!s:5} offset={key[3]!s:5} n={len(groups[key]):4} -> {verdict}"
    )
    for aoi, iid, p1, f in rows:
        print(f"      {aoi} {iid:40} p1={p1:6.0f} frac<1000={f:.3f}")
