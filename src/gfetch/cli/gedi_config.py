import logging
from dataclasses import dataclass, fields
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from gfetch.cli.config import RESERVED_SECTION_KEYS, AOIConfig, TimeRangeConfig, resolve_aoi

log = logging.getLogger(__name__)

__all__ = ["GEDI_PRODUCTS", "GediConfig", "GediL2AConfig", "load"]


@dataclass
class GediConfig:
    """
    GEDI vector-fetch job configuration, as used for L4A (see `GediL2AConfig` for
    L2A's extra fields).

    Its own schema, separate from the raster pipeline's `gfetch.cli.config.Config`
    (see `gfetch gedi`'s standalone-command design in `claude/tech-stack.md`) -
    reuses only the generic `AOIConfig`/`TimeRangeConfig` vocabulary the two
    pipelines actually share.

    Attributes
    ----------
    output : Path
        Output GeoParquet file path. In a unified job config's `gedi_<product>:`
        section, this defaults to `<generic output_dir>/gedi/<product>.parquet` if
        not set explicitly.
    aoi : AOIConfig | None
        Area of interest. Exactly one of `aoi`/`countries` must be given.
    countries : list[str] | None
        Country names to derive the AOI from instead of giving `aoi` directly - see
        `gfetch.cli.config.resolve_aoi`. Exactly one of `aoi`/`countries` must be
        given.
    time_range : TimeRangeConfig | None
        Time range to fetch. Defaults to None, which processes every GEDI
        granule intersecting `aoi` across the whole mission archive
        (2019-present).
    fields : list[str] | None
        Footprint columns to keep from SlideRule's fixed response schema. Defaults
        to None, which uses the product's default (e.g.
        `gfetch.gedi.GEDI_L4A_DEFAULT_FIELDS`).
    anc_fields : list[str] | None
        Extra per-shot fields to read directly out of the source granule, via
        `gfetch.gedi.fetch_gedi_l4a`'s `anc_fields`. Defaults to None (none
        requested).
    quality_filter : bool
        Drop low-quality footprints like geefetch does, via
        `gfetch.gedi.fetch_gedi_l4a`'s `quality_filter`. Defaults to True.
    """

    output: Path
    aoi: AOIConfig | None = None
    countries: list[str] | None = None
    time_range: TimeRangeConfig | None = None
    fields: list[str] | None = None
    anc_fields: list[str] | None = None
    quality_filter: bool = True

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


@dataclass
class GediL2AConfig(GediConfig):
    """
    GEDI L2A vector-fetch job configuration.

    Attributes
    ----------
    rh_percentiles : list[int] | None
        Relative-height percentiles to return as named `rh{p}` columns, via
        `gfetch.gedi.fetch_gedi_l2a`'s `rh_percentiles`. Defaults to None (no
        `rh{p}` columns).
    """

    rh_percentiles: list[int] | None = None


GEDI_PRODUCTS: dict[str, type[GediConfig]] = {"l2a": GediL2AConfig, "l4a": GediConfig}


def load(path: Path, product: str) -> GediConfig:
    """
    Load and validate one GEDI product's job configuration from a unified YAML
    file's `gedi_<product>:` section.

    The file's top level holds generic defaults; only the ones that are also fields
    of the product's config class apply here (plus `output_dir` - a
    raster-pipeline-style base directory `output` defaults from) - a generic field
    that's only meaningful elsewhere (e.g. `n_workers`, or `rh_percentiles` for L4A)
    is ignored, not an error. The reserved `gedi_<product>` section key overrides
    those defaults, and *is* validated strictly (an unknown key there raises). See
    `gfetch.cli.config.load`'s docstring for the full unified-file convention.

    Parameters
    ----------
    path : Path
        Path to the configuration YAML file.
    product : str
        Which GEDI product to load, a key of `GEDI_PRODUCTS`.

    Returns
    -------
    GediConfig
        Fully validated configuration object, a `GediL2AConfig` for `'l2a'`.
        Raises `ValueError` if the file has a bare `gedi:` section, or (from
        `gfetch.cli.config.resolve_aoi`) if neither/both of `aoi`/`countries` are
        given.
    """
    log.debug(f"Loading GEDI {product} config from {path}")
    config_cls = GEDI_PRODUCTS[product]
    raw = OmegaConf.load(path)
    assert isinstance(raw, DictConfig), f"{path} must be a YAML mapping, not a list"
    if "gedi" in raw:
        msg = f"{path}: unknown section `gedi:`, use `gedi_l2a:`/`gedi_l4a:`"
        log.error(msg)
        raise ValueError(msg)
    gedi_field_names = {f.name for f in fields(config_cls)}
    output_dir = raw.get("output_dir")
    generic = OmegaConf.create(
        {k: v for k, v in raw.items() if k in gedi_field_names and k not in RESERVED_SECTION_KEYS}
    )
    section = raw.get(f"gedi_{product}", OmegaConf.create({}))
    assert isinstance(section, DictConfig)

    overrides = OmegaConf.merge(generic, section)
    assert isinstance(overrides, DictConfig)
    if "output" not in section and output_dir is not None:
        overrides["output"] = str(Path(str(output_dir)) / "gedi" / f"{product}.parquet")

    structured = OmegaConf.structured(config_cls)
    merged = OmegaConf.merge(structured, overrides)
    OmegaConf.resolve(merged)
    cfg: GediConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]
    cfg.aoi = resolve_aoi(cfg.aoi, cfg.countries, context=str(path))
    log.debug(f"Resolved output={cfg.output}")
    return cfg
