import warnings; warnings.filterwarnings("ignore")
from collections import defaultdict
import numpy as np, rasterio
from rasterio.windows import Window
import pystac_client
c = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
bbox=[34.5,-14.5,36.5,-12.5]
l2a = list(c.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime="2021-12-01/2022-06-01").items())
seen=set()
for i in l2a:
    k=(i.properties["s2:processing_baseline"], i.properties.get("earthsearch:boa_offset_applied"), i.assets["red"].extra_fields["raster:bands"][0].get("offset"))
    if k not in seen: seen.add(k); print("l2a baseline/boa_offset_applied/red offset:", k)
c1 = list(c.search(collections=["sentinel-2-c1-l2a"], bbox=bbox, datetime="2021-12-01/2021-12-31").items())
print("c1 Dec21 props sample:", {k:v for k,v in c1[0].properties.items() if "offset" in k or "baseline" in k})
g=defaultdict(dict)
for i in l2a:
    g[(i.properties["grid:code"], i.datetime.date())]["l2a-"+i.properties["s2:processing_baseline"]]=i
for i in c1:
    g[(i.properties["grid:code"], i.datetime.date())]["c1-"+i.properties["s2:processing_baseline"]]=i
trip=[(k,v) for k,v in g.items() if len(v)>=3]
trip.sort(key=lambda kv: min(x.properties["eo:cloud_cover"] for x in kv[1].values()))
(k,v)=trip[0]; print("scene", k, {n:x.properties["eo:cloud_cover"] for n,x in v.items()})
ref = next(x for n,x in v.items() if n.startswith("c1"))
with rasterio.open(ref.assets["scl"].href) as s: scl=s.read(1, window=Window(1000,1000,900,900))
m20 = np.isin(scl,[4,5])
for b in ["blue","red","nir","swir16"]:
    out=[]
    for n,x in sorted(v.items()):
        with rasterio.open(x.assets[b].href) as src:
            f=10/src.res[0]; a=src.read(1, window=Window(2000*f,2000*f,1800*f,1800*f)).astype(float)
        m = np.kron(m20, np.ones((2,2),bool)) if a.shape[0]==1800 else m20
        out.append(f"{n}: median DN {np.median(a[m & (a>0)]):.0f}")
    print(b, " | ".join(out))
