# Sentinel-2 DN harmonization (design, 2026-10-16)

Goal: Sentinel-2 composites from 2017 to the present with one DN encoding throughout:
Collection 1's, `DN = reflectance × 10000 + 1000`. Evidence for every rule below is in
`claude/s2-2022/README.md` (Earth Search) and `claude/s2-2022/enc_pc.py` (Planetary
Computer).

**Revisit once Earth Search v2 leaves beta.** v2's `sentinel-2-l2a` replaces v1's C1,
drops baseline 05.09 (a Dec 2022 - Nov 2023 hole), and may come with a reprocessed
history. The gaps, fallbacks and encoding rules here are all specific to v1 as of
2026-10.

## Search: collection resolution

- 2015-06-23 to 2019-03-21 gains a fallback to v1 `sentinel-2-l2a`, like 2022.
  `_CollectionGap` gets a `confirm: bool` field. On a confirming gap, `search` logs a
  loud warning (heterogeneous processing: baselines 00.01 to 03.0x, older Sen2Cor
  versions, probably no GRI geometric refinement before 03.00), then:
  - asks for a y/N confirmation on a TTY,
  - with the `--accept-heterogeneous-baselines` flag on `gfetch <sat> search`, proceeds
    without asking,
  - errors out on a non-TTY without the flag, since a batch job can't answer a prompt.
- A range straddling 2019-03-21/22, or either 2022 boundary, is refused. This is
  `resolve_collection`'s existing "not entirely within" rule, unchanged.
- The prompt lives in the CLI layer (`gfetch.cli.search`). `gfetch.search.search()`
  takes an `accept_heterogeneous: bool = False` and raises without it.

## Per-item encoding: `gfetch:dn_offset`

Every Sentinel-2 item gets an integer property, `gfetch:dn_offset` ∈ {0, 1000}: the
offset its DNs carry. A second property, `gfetch:dn_offset_from` ∈ {`metadata`,
`pixels`, `date`}, records how it was decided.

| Source / collection | Rule |
|---|---|
| earthsearch `sentinel-2-c1-l2a` | 1000 |
| earthsearch `sentinel-2-l2a`, acquired 2022-01-25..2022-02-27, `boa_offset_applied: false`, baseline ≥ 04.00 | pixel test; 1000 if inconclusive (`date`) |
| earthsearch `sentinel-2-l2a`, everything else | 0, whatever `boa_offset_applied` and `raster:bands` say |
| planetary-computer `sentinel-2-l2a` | 1000 if baseline ≥ 04.00, else 0 |
| anything else (`custom` satellites) | not set; no harmonization |

Pixel test: the fraction of blue DNs below 1000 over SCL 4/6 (vegetation, water), read
at about 1/20 resolution.
- ≤ 0.02 → 1000; ≥ 0.5 → 0.
- Fewer than 2000 such pixels, or a fraction in between → inconclusive, so the date
  rule applies.
- It's a `gfetch.dn` module, so `search`, `download` and `mosaic` can share it.

`search` logs a warning whenever any item went through the window rule: "DN offset
decided by a pixel heuristic for N item(s) acquired 2022-01-25..02-27 (M by date
alone)".

**Where the pixel test runs (decided 2026-10-16): once, in `search`, on the remote files.**
- One code path, persisted in `items.json`, which every later stage already reads.
  `download` deep-copies items, so the properties reach the cached `items.json`.
- Cost: two small remote reads per window item, only for ranges touching that window.
- Rejected alternative: `download` on the local files, plus remote-mode `mosaic` in
  memory. That's two call sites, and every concurrent `mosaic` task would repeat the
  reads.

**Implemented 2026-10-16** (`gfetch.dn`). Live check over 0.2° in Missouri, 1 Feb -
10 Mar 2022:
- 60 items: 14 classified +1000 from pixels, 26 +1000 by date, 20 no offset from
  metadata.
- Per item, SCL 4/6 blue DNs: 1 Feb - 19 Feb has median ~1700 (p10 ~1480); 27 Feb
  onward has ~400-600.
- The date-assigned items share days with pixel-classified +1000 neighbours. Their few
  "clear" pixels are hazy.

## Mosaic and OCM: applying it

- odc-stac's `groupby="solar_day"` fuses same-day items into one time step before gfetch
  sees pixels. A per-item shift therefore has to happen before loading, per encoding
  group.
- `mosaic` partitions the items by `gfetch:dn_offset`, the same way
  `_composite_by_orbit_state` partitions by orbit state, and loads each group
  separately. After `mask_nodata`, it adds `1000 - offset` to the data bands (never the
  mask band), concatenates the groups along `time`, then masks clouds and composites.
  - One load when every item agrees, which is the common case.
  - Only a tile overlap on a day whose two tiles differ in encoding gets two time steps
    instead of one. That's negligible, and the values are equal after the shift.
- `ocm.read_input` adds `1000 - offset` to the non-zero DNs before `predict`, so the
  model always sees C1 encoding.
- Items without `gfetch:dn_offset` (`custom` satellites, or `items.json` written before
  this change) aren't shifted. For built-in `sentinel-2`, `mosaic` refuses them and asks
  for a re-run of `search`, to avoid silently mixing encodings.
- OmniCloudMask masks already cached from unshifted inputs aren't recomputed. After
  re-running `search` on an old job, delete its cached OCM masks too.

## Out of scope, noted

- PC dedup: PC lists reprocessed duplicates (e.g. 2022-07-24 at both 04.00 and 05.10).
  Both carry +1000, so encoding isn't affected. Since 2026-10-16,
  `_dedupe_sentinel2_processing_baseline` collapses them via `s2:mgrs_tile`.
- Non-encoding differences between baselines (algorithm, SCL, geolocation) aren't
  corrected. 04.00 vs 05.10 measured ≤ 0.005 mean |diff|; 00.01 hasn't been measured.
