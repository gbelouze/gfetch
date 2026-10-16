# S2 for 2022: open investigation (started 2026-09-29)

Goal: decide how to get Sentinel-2 for the Mozambania 2022 config
(`~/Documents/jz/src/configs/mozambania/v6/2022/download_gfetch/config.yaml`) and how
different it will be from the other years (Collection 1, `sentinel-2-c1-l2a`).

## Established

- The config's `2021-12-01 → 2022-06-01` straddles the Earth Search C1 2022 gap, so
  `resolve_collection` raises. The 2023 config (`2022-12-01 → 2023-06-01`) does too.
- Tentative plan: per-section `s2.time_range`: 2022 → `2022-01-25 → 2022-06-01`
  (falls back to `sentinel-2-l2a`, skips baseline 03.01), 2023 → `2023-01-01 →
  2023-06-01` (C1). Not applied to the configs yet: it depends on the encoding
  question below.
- Baselines over northern Mozambique (bbox 34.5,-14.5,36.5,-12.5; `bl.py`):
  - `sentinel-2-l2a`, Dec 2021 – May 2022: 03.01 in Dec and up to 24 Jan; 04.00 from
    25 Jan; plus reprocessed duplicates (05.00 in Dec 2021, 05.10 on a few Mar/Apr
    2022 dates). The per-(tile, date) dedup keeps the highest baseline.
  - C1: 2021 is all 05.00, Jan–May 2023 is all 05.09, and 2024 is 05.10.
- Reprocessing alone changes little. 04.00 vs 05.10 on the same tile/date in
  `sentinel-2-l2a` (6 pairs, clear pixels per SCL 4/5; `pairs.py`): mean bias
  ≤ 0.001 reflectance, mean |diff| 0.001–0.005 (largest in nir/rededge), per-scene
  corr ≥ 0.99, in all 12 bands.

## Open: DN encoding is not what gfetch assumes

`sources.py`/`tech-stack.md` claim that `sentinel-2-l2a` items from 04.00 on carry the
+1000 DN offset like C1, and 03.01 items don't. The data contradicts this:

- `earthsearch:boa_offset_applied` is mixed within the same baseline (`off.py`):
  04.00 True *and* False, 05.00 True, 05.10 False, 03.01 False. `raster:bands` offset
  is -0.1 on every one of those except 03.01 (0).
- On tile 36LZM (`dn.py`), 05.00 `applied=True` has the same DNs as 03.01 on the same
  date (2021-10-15: blue 612 vs 619, red 1130 vs 1136). So "applied" seems to mean
  Earth Search *subtracted* the offset, leaving no +1000, while the item still
  advertises offset -0.1.
- `pairs.py`: using the -0.1 offset, the 04.00 items average *negative* blue/red
  over clear land (≈ -0.035), consistent with the offset already having been removed.

### Resolved 2026-10-16: per-item DN encoding (`enc.py`, `enc2.py`)

Method: per item, the fraction of blue DNs below 1000 over SCL-clear vegetation/water
(SCL 4/6). Their blue reflectance is ~0.01-0.05, so it's ~1 without the offset and ~0
with it. Clouds must be masked first: unmasked, cloudy items read as +1000 whatever
their encoding. Four boxes (France 1.5..4.5°E 46.5..48.5°N, Gorongosa 34..35°E
19.5..18.5°S, Kanto 138.5..140.5°E 35..36.5°N, Missouri 91..89°W 38..39.5°N).
The fraction is bimodal: of ~1,000 classified items, 738 are ≤ 0.01 and ~800 are ≥ 0.5.

- `enc.py`: 8 items (cloud cover < 30%) per (collection, baseline,
  `boa_offset_applied`, `raster:bands` offset) group, 2017-2026.
  - C1, every baseline 05.00-05.12 (no `boa_offset_applied`, offset -0.1): +1000.
  - `sentinel-2-l2a`, baselines 00.01, 02.08-02.14, 03.01 (offset 0): no offset.
  - `sentinel-2-l2a`, 04.00-05.12 with `applied: true` (offset -0.1): no offset.
    `raster:bands` is wrong for these: a client applying it subtracts 0.1 again.
- `enc2.py`: *every* `sentinel-2-l2a` item with `applied: false` acquired from
  2022-01-25 on, in the four boxes: 1,537 items, 576 too cloudy to classify.
  - Acquired 2022-01-25 to 2022-02-27, baseline 04.00, created 2022-11-03/06: 168 +1000,
    8 none (25-30 Jan).
  - Every other one: no offset. That's 285 items from Mar-Nov 2022 (04.00, 05.10) and
    500 from Nov 2024-Jul 2026 (05.11, 05.12).
- So `applied: false` is wrong for most items, and `raster:bands` offset is wrong for
  every no-offset item from 04.00 on. Neither metadata field determines the encoding.
  The only +1000 items in `sentinel-2-l2a` are acquired 2022-01-25 to 2022-02-27, and
  not all items acquired then.
- Consequences:
  - A gfetch fallback to `sentinel-2-l2a` is uniformly no-offset, except that window.
  - It is 1000 DN below C1 everywhere.
  - A ≤ 2019 fallback is internally consistent too (00.01, 02.xx, and the 05.00
    `applied: true` duplicates are all no-offset).

Old to-do list (items 1-3 answered above):
1. Let `dn.py` finish. Compare raw DNs on the same tile across `sentinel-2-l2a`
   2022 (04.00 True/False, 05.10) and C1 2021/2023, same season. Does each group sit
   near C1 (+1000) or near 03.01 (no offset)?
2. If the encoding is mixed within 2022, gfetch has to harmonize per item using
   `boa_offset_applied`, or the median mixes encodings. That would be a gfetch fix, and
   the caveat in `sources.py` and the 2022 paragraph of `tech-stack.md` would need
   correcting.
3. Also check C1's own DN level: the C1 items above have no `boa_offset_applied`
   property.
4. Then apply the `s2.time_range` edits to the 2022/2023 configs.

Scripts in this directory run with `uv run python -u claude/s2-2022/<script>.py`.
Each hits Earth Search S3 anonymously, and `pairs.py`/`dn.py` take about 10 minutes.
