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

To do next:
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
