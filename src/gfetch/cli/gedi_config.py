import logging
from dataclasses import dataclass, fields
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from gfetch.cli.config import RESERVED_SECTION_KEYS, AOIConfig, TimeRangeConfig, resolve_aoi

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
    output : Path
        Output GeoParquet file path. In a unified job config's `gedi:` section, this
        defaults to `<generic output_dir>/gedi/l2a.parquet` if not set explicitly.
    aoi : AOIConfig | None
        Area of interest. Exactly one of `aoi`/`countries` must be given.
    countries : list[str] | None
        Country names to derive the AOI from instead of giving `aoi` directly - see
        `gfetch.cli.config.resolve_aoi`. Exactly one of `aoi`/`countries` must be
        given.
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

    output: Path
    aoi: AOIConfig | None = None
    countries: list[str] | None = None
    time_range: TimeRangeConfig | None = None
    fields: list[str] | None = None
    anc_fields: list[str] | None = None
    rh_percentiles: list[int] | None = None

    def __post_init__(self) -> None:
        self.output = Path(self.output).expanduser().absolute()

    @property
    def resolved_aoi(self) -> AOIConfig:
        """
        Returns
        -------
        AOIConfig
            `aoi`, resolved by `load()` (from `countries`, if `aoi` wasn't given
            directly) - never None on a `GediConfig` returned by `load()`. `aoi`
            itself stays `AOIConfig | None` at the schema level so OmegaConf
            accepts a config that gives `countries` instead.
        """
        assert self.aoi is not None, "aoi not yet resolved - build this GediConfig via load()"
        return self.aoi


def load(path: Path) -> GediConfig:
    """
    Load and validate a GEDI job configuration from a unified YAML file's `gedi:`
    section.

    The file's top level holds generic defaults; only the ones that are also
    `GediConfig` fields apply here (plus `output_dir` - a raster-pipeline-style base
    directory `output` defaults from) - a generic field that's only meaningful for
    the raster pipeline (e.g. `n_workers`) is ignored, not an error. The reserved
    `gedi` section key overrides those defaults, and *is* validated strictly (an
    unknown key there raises). See `gfetch.cli.config.load`'s docstring for the full
    unified-file convention (other reserved sections - `s1`/`s2`/`custom` - are
    irrelevant here).

    Parameters
    ----------
    path : Path
        Path to the configuration YAML file.

    Returns
    -------
    GediConfig
        Fully validated configuration object. Raises `ValueError` (from
        `gfetch.cli.config.resolve_aoi`) if neither/both of `aoi`/`countries` are
        given.
    """
    log.debug(f"Loading GEDI config from {path}")
    raw = OmegaConf.load(path)
    assert isinstance(raw, DictConfig), f"{path} must be a YAML mapping, not a list"
    gedi_field_names = {f.name for f in fields(GediConfig)}
    output_dir = raw.get("output_dir")
    generic = OmegaConf.create(
        {k: v for k, v in raw.items() if k in gedi_field_names and k not in RESERVED_SECTION_KEYS}
    )
    section = raw.get("gedi", OmegaConf.create({}))
    assert isinstance(section, DictConfig)

    overrides = OmegaConf.merge(generic, section)
    assert isinstance(overrides, DictConfig)
    if "output" not in section and output_dir is not None:
        overrides["output"] = str(Path(str(output_dir)) / "gedi" / "l2a.parquet")

    structured = OmegaConf.structured(GediConfig)
    merged = OmegaConf.merge(structured, overrides)
    OmegaConf.resolve(merged)
    cfg: GediConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]
    cfg.aoi = resolve_aoi(cfg.aoi, cfg.countries, context=str(path))
    log.debug(f"Resolved output={cfg.output}")
    return cfg
