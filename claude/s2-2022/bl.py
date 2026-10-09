from collections import Counter
import pystac_client
c = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
bbox = [34.5, -14.5, 36.5, -12.5]  # northern Mozambique
for coll, dt in [
    ("sentinel-2-l2a", "2021-12-01/2022-06-01"),
    ("sentinel-2-c1-l2a", "2020-12-01/2021-06-01"),
    ("sentinel-2-c1-l2a", "2023-01-01/2023-06-01"),
    ("sentinel-2-c1-l2a", "2023-12-01/2024-06-01"),
]:
    items = list(c.search(collections=[coll], bbox=bbox, datetime=dt, max_items=5000).items())
    bl = Counter((i.properties.get("s2:processing_baseline"), i.datetime.strftime("%Y-%m") ) for i in items)
    print(coll, dt, len(items))
    for k in sorted(bl): print("   ", k, bl[k])
    if items:
        a = items[-1].assets["red"].extra_fields.get("raster:bands")
        print("    red raster:bands", a, "boa_offset_applied", items[-1].properties.get("earthsearch:boa_offset_applied"))
