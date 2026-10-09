import warnings; warnings.filterwarnings("ignore")
import numpy as np, rasterio
from rasterio.windows import Window
import pystac_client
c = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
tile="36LZM"
q={"grid:code":{"eq":f"MGRS-{tile}"},"eo:cloud_cover":{"lt":10}}
runs=[("sentinel-2-l2a","2021-10-01/2022-10-31"),("sentinel-2-c1-l2a","2021-01-01/2021-10-31"),("sentinel-2-c1-l2a","2023-01-01/2023-10-31")]
for coll,dt in runs:
    for i in sorted(c.search(collections=[coll], datetime=dt, query=q).items(), key=lambda i:i.datetime):
        with rasterio.open(i.assets["scl"].href) as s: scl=s.read(1, window=Window(2000,2000,1000,1000))
        m=np.kron(np.isin(scl,[4,5]),np.ones((2,2),bool))
        if m.mean()<0.3: continue
        r=[]
        for b in ["blue","red","nir"]:
            with rasterio.open(i.assets[b].href) as src: a=src.read(1, window=Window(4000,4000,2000,2000)).astype(float)
            r.append(np.median(a[m&(a>0)]))
        print(f"{coll:18} {i.datetime.date()} bl={i.properties['s2:processing_baseline']} applied={i.properties.get('earthsearch:boa_offset_applied')!s:5} cc={i.properties['eo:cloud_cover']:5.1f} blue={r[0]:5.0f} red={r[1]:5.0f} nir={r[2]:5.0f}")
