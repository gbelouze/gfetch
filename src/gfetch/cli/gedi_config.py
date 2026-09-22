import logging
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from gfetch.cli.config import AOIConfig, TimeRangeConfig

log = logging.getLogger(__name__)

__all__ = ["GediConfig", "load"]


@dataclass
class GediConfig:
    """
    GEDI L2A vector-fetch job configuration.

    Its own schema, separate from the raster pipeline's `gfetch.cli.config.Config`
    (see `gfetch gedi`'s standalone-command design in `claude/tech-stack.md`) -
    reuses only the generic `AOIConfig`/`TimeRangeConfig` vocabulary the two
    pipelines actually share.

    Attributes
    ----------
    aoi : AOIConfig
        Area of interest.
    output : Path
        Output GeoParquet file path.
    time_range : TimeRangeConfig | None
        Time range to fetch. Defaults to None, which processes every GEDI L2A
        granule intersecting `aoi` across the whole mission archive
        (2019-present).
    fields : list[str] | None
        Footprint columns to keep from SlideRule's fixed `gedi02ap` schema.
        Defaults to None, which uses `gfetch.gedi.GEDI_L2A_DEFAULT_FIELDS`.
    anc_fields : list[str] | None
        Extra per-shot fields to read directly out of the source granule, via
        `gfetch.gedi.fetch_gedi_l2a`'s `anc_fields`. Defaults to None (none
        requested).
    rh_percentiles : list[int] | None
        Relative-height percentiles to pull out of the `rh` ancillary field into
        named `rh{p}` columns, via `gfetch.gedi.expand_rh`. Defaults to None,
        which leaves `rh` as a raw 101-element array column (or omits it, if not
        requested via `anc_fields`).
    """

    aoi: AOIConfig
    output: Path
    time_range: TimeRangeConfig | None = None
    fields: list[str] | None = None
    anc_fields: list[str] | None = None
    rh_percentiles: list[int] | None = None

    def __post_init__(self) -> None:
        self.output = Path(self.output).expanduser().absolute()


def load(path: Path) -> GediConfig:
    """
    Load and validate a GEDI job configuration from a YAML file.

    Parameters
    ----------
    path : Path
        Path to the configuration YAML file.

    Returns
    -------
    GediConfig
        Fully validated configuration object.
    """
    log.debug(f"Loading GEDI config from {path}")
    from_yaml = OmegaConf.load(path)
    structured = OmegaConf.structured(GediConfig)
    merged = OmegaConf.merge(structured, from_yaml)
    OmegaConf.resolve(merged)
    cfg: GediConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]
    log.debug(f"Resolved output={cfg.output}")
    return cfg
