import logging
from pathlib import Path

from gfetch.cli.gedi_config import load
from gfetch.gedi import GEDI_L2A_DEFAULT_FIELDS, expand_rh, fetch_gedi_l2a
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

    Parameters
    ----------
    config_path : Path
        Path to the configuration YAML file.
    """
    cfg = load(config_path)

    anc_fields = list(cfg.anc_fields) if cfg.anc_fields is not None else []
    if cfg.rh_percentiles is not None and "rh" not in anc_fields:
        anc_fields.append("rh")

    start = cfg.time_range.start if cfg.time_range is not None else None
    end = cfg.time_range.end if cfg.time_range is not None else None
    time_range = _to_time_range(start, end)
    with count_bar() as progress:
        gdf = fetch_gedi_l2a(
            cfg.resolved_aoi.bbox,
            time_range=time_range,
            fields=cfg.fields if cfg.fields is not None else GEDI_L2A_DEFAULT_FIELDS,
            anc_fields=anc_fields or None,
            progress=progress,
        )
    if cfg.rh_percentiles is not None:
        gdf = expand_rh(gdf, cfg.rh_percentiles)

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(cfg.output)
    log.info(f"Wrote {len(gdf)} footprint(s) to {cfg.output}")
