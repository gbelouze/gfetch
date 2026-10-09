import warnings; warnings.filterwarnings("ignore")
from collections import defaultdict
import numpy as np, rasterio
from rasterio.windows import Window
import pystac_client
c = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
items = list(c.search(collections=["sentinel-2-l2a"], bbox=[34.5,-14.5,36.5,-12.5], datetime="2022-01-25/2022-06-01").items())
g = defaultdict(dict)
for i in items:
    g[(i.properties["grid:code"], i.datetime.date())][i.properties["s2:processing_baseline"]] = i
pairs = [(k, v) for k, v in g.items() if "04.00" in v and "05.10" in v]
print(len(pairs), "pairs")
bands = ["blue","green","red","rededge1","rededge2","rededge3","nir","nir08","swir16","swir22","coastal","nir09"]
res = defaultdict(list)
for (tile, d), v in sorted(pairs, key=lambda p: p[1]["05.10"].properties["eo:cloud_cover"])[:6]:
    old, new = v["04.00"], v["05.10"]
    print(tile, d, "cloud%", old.properties.get("eo:cloud_cover"), new.properties.get("eo:cloud_cover"))
    with rasterio.open(new.assets["scl"].href) as s:
        scl = s.read(1, window=Window(1000, 1000, 900, 900))  # 20 m
    for b in bands:
        rows = []
        for it in (old, new):
            with rasterio.open(it.assets[b].href) as src:
                f = 10 / src.res[0]
                w = Window(2000*f, 2000*f, 1800*f, 1800*f)
                rows.append((src.read(1, window=w).astype(np.float64), src.transform))
        a, bb = rows[0][0], rows[1][0]
        # bring SCL mask to band resolution
        m = scl == 4  # vegetation, clear
        m = m | (scl == 5)
        rep = a.shape[0] / m.shape[0]
        if rep >= 1:
            m = np.kron(m, np.ones((int(rep), int(rep)), bool))
        else:
            k = int(1/rep); m = m.reshape(a.shape[0], k, a.shape[1], k).all((1,3))
        m &= (a > 0) & (bb > 0)
        if m.sum() < 1000: continue
        da = (bb[m] - a[m]) * 1e-4
        res[b].append((m.sum(), np.mean(da), np.mean(np.abs(da)), np.corrcoef(a[m], bb[m])[0,1], np.mean(a[m])*1e-4-0.1))
print(f"{'band':9} {'n_pairs':>7} {'mean diff':>10} {'mean |diff|':>11} {'corr':>6} {'mean refl':>9}")
for b in bands:
    r = np.array(res[b])
    if len(r)==0: print(b, "no valid"); continue
    print(f"{b:9} {len(r):7d} {r[:,1].mean():+10.4f} {r[:,2].mean():11.4f} {r[:,3].mean():6.3f} {r[:,4].mean():9.3f}")
