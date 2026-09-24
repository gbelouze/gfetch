import logging
import shutil
from pathlib import Path

from gfetch.cli.gedi_config import load
from gfetch.gedi import GEDI_L2A_DEFAULT_FIELDS, fetch_gedi_l2a, write_geoparquet
from gfetch.utils.progress import count_bar

log = logging.getLogger(__name__)


def _to_time_range(start: str | None, end: str | None) -> tuple[str, str] | None:
    """
    Translate an optional config `TimeRangeConfig`'s dates into a SlideRule
    `(t0, t1)` time range.

    Parameters
    ----------
    start : str | None
        Start date (`'YYYY-MM-DD'`), inclusive.
    end : str | None
        End date (`'YYYY-MM-DD'`), inclusive.

    Returns
    -------
    tuple[str, str] | None
        `(t0, t1)` in SlideRule's UTC timestamp format, or None if both `start` and
        `end` are None.
    """
    if start is None and end is None:
        return None
    t0 = f"{start}T00:00:00Z" if start is not None else "1900-01-01T00:00:00Z"
    t1 = f"{end}T23:59:59Z" if end is not None else "2100-01-01T00:00:00Z"
    return (t0, t1)


def gedi(config_path: Path) -> None:
    """
    Fetch GEDI L2A footprints matching a configuration's AOI/time range and write
    them to GeoParquet.

    Resumable: each tile's footprints are saved under `<output>.tiles/` as soon as
    they're fetched, and a rerun reuses them. That directory is removed once the
    output is written.

    Parameters
    ----------
    config_path : Path
        Path to the configuration YAML file.
    """
    cfg = load(config_path)

    start = cfg.time_range.start if cfg.time_range is not None else None
    end = cfg.time_range.end if cfg.time_range is not None else None
    time_range = _to_time_range(start, end)
    tile_dir = cfg.output.with_name(f"{cfg.output.name}.tiles")

    polygon = None
    if cfg.countries is not None:
        from gfetch.countries import resolve_country_polygon

        polygon = resolve_country_polygon(cfg.countries)

    with count_bar() as progress:
        gdf = fetch_gedi_l2a(
            cfg.resolved_aoi.bbox,
            time_range=time_range,
            fields=cfg.fields if cfg.fields is not None else GEDI_L2A_DEFAULT_FIELDS,
            anc_fields=cfg.anc_fields,
            polygon=polygon,
            rh_percentiles=cfg.rh_percentiles,
            quality_filter=cfg.quality_filter,
            tile_dir=tile_dir,
            progress=progress,
        )
    write_geoparquet(gdf, cfg.output)
    log.info(f"Wrote {len(gdf)} footprint(s) to {cfg.output}")
    if tile_dir.exists():
        shutil.rmtree(tile_dir)
    log.debug(f"Removed per-tile results in {tile_dir}")
