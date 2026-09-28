import logging
from dataclasses import dataclass, field, fields
from functools import cached_property
from pathlib import Path
from typing import Any

import shapely
from odc.geo.crs import CRS
from omegaconf import DictConfig, OmegaConf
from shapely.geometry.base import BaseGeometry

from gfetch.profiles import OCM_BAND, OCM_INPUT_BANDS, OCM_MASK_OUT, get_profile
from gfetch.utils.system import available_cpus

log = logging.getLogger(__name__)

__all__ = [
    "AOIConfig",
    "BUILTIN_SATELLITES",
    "ORBIT_STATE_AS_BANDS",
    "Config",
    "TimeRangeConfig",
    "load",
    "resolve_aoi",
    "resolve_aoi_geometry",
    "resolve_bands",
    "resolve_cloud_mask",
    "resolve_output_variables",
    "resolve_source",
]

ORBIT_STATE_AS_BANDS = "as_bands"

# Section keys reserved in a unified job config file (see `load`'s docstring) - not
# valid `Config` field names, so they're stripped out of the "generic" dict before
# merging a satellite's section over it.
RESERVED_SECTION_KEYS = frozenset({"s1", "s2", "gedi_l2a", "gedi_l4a", "custom"})

# CLI satellite key -> canonical `gfetch.profiles`/`gfetch.sources` satellite name,
# force-set by `load` regardless of what a config's section itself contains.
BUILTIN_SATELLITES: dict[str, str] = {"s1": "sentinel-1", "s2": "sentinel-2"}


@dataclass
class AOIConfig:
    """
    Area of Interest configuration, as a bounding box in EPSG:4326.

    Attributes
    ----------
    left : float
        Left (west) bound, in decimal degrees.
    bottom : float
        Bottom (south) bound, in decimal degrees.
    right : float
        Right (east) bound, in decimal degrees.
    top : float
        Top (north) bound, in decimal degrees.
    """

    left: float
    bottom: float
    right: float
    top: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """
        Returns
        -------
        tuple[float, float, float, float]
            The AOI as (min_lon, min_lat, max_lon, max_lat).
        """
        return (self.left, self.bottom, self.right, self.top)


def resolve_aoi(aoi: AOIConfig | None, countries: list[str] | None, *, context: str) -> AOIConfig:
    """
    Resolve a job's effective AOI: either `aoi` as given, or the bounding box of
    the union of `countries`' boundaries (via `gfetch.countries.
    resolve_country_polygon`) - mirrors geefetch's `aoi.country` field, which is
    likewise resolved to a bounding box for the actual data request (its own use of
    the exact polygon shape is a coarse, tile-level filter downstream, never a
    per-pixel clip). Exactly one of `aoi`/`countries` must be given.

    Parameters
    ----------
    aoi : AOIConfig | None
        Explicit AOI, if given directly.
    countries : list[str] | None
        Country names to derive the AOI from, if `aoi` wasn't given directly.
    context : str
        Description of what's being resolved (e.g. the config path), used only to
        make the "exactly one of" error message identify which load failed.

    Returns
    -------
    AOIConfig
        `aoi` unchanged, or the bounding box of `countries`' union.

    Raises
    ------
    ValueError
        If both or neither of `aoi`/`countries` are given.
    """
    if (aoi is None) == (countries is None):
        raise ValueError(f"{context}: exactly one of `aoi` or `countries` must be given.")
    if aoi is not None:
        return aoi
    assert countries is not None
    from gfetch.countries import resolve_country_polygon

    min_lon, min_lat, max_lon, max_lat = resolve_country_polygon(countries).bounds
    return AOIConfig(left=min_lon, bottom=min_lat, right=max_lon, top=max_lat)


def resolve_aoi_geometry(aoi: AOIConfig, countries: list[str] | None) -> BaseGeometry:
    """
    Resolve a job's exact AOI shape, for anything selecting data (search, GEDI tiles,
    mosaic shards). Output grids use the bounding box `aoi` instead.

    Parameters
    ----------
    aoi : AOIConfig
        The job's resolved AOI bounding box, see `resolve_aoi`.
    countries : list[str] | None
        Country names the AOI was derived from, if any.

    Returns
    -------
    BaseGeometry
        The union of `countries`' boundaries, or `aoi` as a box if `countries` is
        None, in EPSG:4326.
    """
    if countries is None:
        return shapely.box(*aoi.bbox)
    from gfetch.countries import resolve_country_polygon

    return resolve_country_polygon(countries)


@dataclass
class TimeRangeConfig:
    """
    Time range configuration for filtering STAC search results.

    Attributes
    ----------
    start : str
        Start date, in 'YYYY-MM-DD' format.
    end : str
        End date, in 'YYYY-MM-DD' format.
    """

    start: str
    end: str

    @property
    def datetime(self) -> str:
        """
        Returns
        -------
        str
            The time range in STAC API datetime-interval format.
        """
        return f"{self.start}/{self.end}"


@dataclass
class Config:
    """
    Main gfetch job configuration.

    Attributes
    ----------
    time_range : TimeRangeConfig
        Time range to search.
    output_dir : Path
        Directory holding the search/download hand-off files, the downloaded asset
        cache, and the output Zarr store.
    aoi : AOIConfig | None
        Area of interest. Exactly one of `aoi`/`countries` must be given.
    countries : list[str] | None
        Country names to derive the AOI from instead of giving `aoi` directly - the
        AOI becomes the bounding box of their union (see `gfetch.countries.
        resolve_country_polygon` for name matching). Exactly one of `aoi`/
        `countries` must be given.
    satellite : str
        Satellite/profile name (e.g. 'sentinel-2'). Defaults to 'sentinel-2'.
    source : str | None
        STAC source name (e.g. 'earthsearch', 'planetary-computer'). Defaults to
        None, see `resolve_source`.
    bands : list[str] | None
        Asset keys to download/load. Defaults to None, which uses the satellite
        profile's default bands.
    max_cloud_cover : float | None
        Maximum `eo:cloud_cover` percentage to search for. Defaults to None (no
        filter). Only meaningful for optical satellites (e.g. Sentinel-2).
    orbit_state : str | None
        Restrict the search to one `sat:orbit_state` ('ascending' or 'descending'),
        e.g. to avoid blending SAR backscatter from different look geometries into one
        composite. 'as_bands' searches both and composites each separately, into
        `{band}_ascending`/`{band}_descending` variables (see
        `resolve_output_variables`). Defaults to None (no filter, both orbit states
        composited together).
    collection : str | None
        Explicit STAC collection id, used verbatim instead of resolving `satellite`
        through `gfetch.sources.StacSource.collection`. Required for a `custom`
        satellite section (there's no registry entry to resolve from); ignored for a
        built-in (`s1`/`s2`) satellite. Defaults to None.
    cloud_mask_band : str | None
        Asset key of the cloud-mask classification band, overriding
        `gfetch.profiles.SatelliteProfile.cloud_mask_band`. Required for a `custom`
        satellite that wants cloud masking (there's no profile to default from);
        optional for a built-in satellite, where it overrides the profile's own
        band. Defaults to None (no override - built-ins keep their profile's band,
        custom satellites get no masking).
    cloud_mask_out : list[int]
        Classification values to mask out, overriding
        `gfetch.profiles.SatelliteProfile.cloud_mask_out`. Only meaningful together
        with `cloud_mask_band`. Defaults to an empty list (no override).
    ocm : bool
        Cloud-mask with OmniCloudMask instead of SCL (Sentinel-2 only): `download`
        also fetches the model weights, `gfetch s2 ocm` computes each item's mask,
        and `mosaic` masks out thick cloud, thin cloud and cloud shadow with it.
        `bands` must include 'red', 'green' and 'nir', and `cloud_mask_band` must
        be unset. Defaults to False.
    ocm_model_dir : Path | None
        Directory holding OmniCloudMask's model weights, shared by the `download`
        and `ocm` stages. Defaults to None, see `ocm_model_path`.
    resolution : float
        Output pixel resolution, in the target CRS's units (meters, for UTM).
        Defaults to 10.0.
    n_workers : int
        Maximum number of items downloading concurrently. Defaults to 4.
    resampling : dict[str, str]
        Per-band resampling method, e.g. `{"nir09": "bilinear"}`. A `"*"` key sets the
        default for bands not otherwise listed. The satellite profile's cloud-mask
        band, if any, is always loaded with 'nearest' resampling regardless of this
        setting. Defaults to an empty dict, which uses odc-stac's own default
        ('nearest' for every band).
    n_compute_workers : Any
        Number of dask threads used to compute and write the `mosaic` stage's
        output: an int, or a `[min, max]` range that `mosaic` tunes within as it
        goes (see `gfetch.utils.tuning.WorkerTuner`). The range's max must be
        memory-safe: more threads means more chunks in memory at once. Defaults to
        None, which uses `gfetch.utils.system.available_cpus` (the CPUs actually
        reserved for this process, e.g. by a SLURM job's `--cpus-per-task`, not the
        whole node).
    chunks : dict[str, int] | None
        Dask chunk sizes for the `mosaic` stage's load step, e.g. `{"x": 512, "y":
        512}`. Smaller spatial chunks bound the memory a single output chunk's
        `gfetch.mosaic.composite` reduction needs: a non-associative reduction (e.g.
        the default 'median') must gather its whole `time` axis into memory per
        spatial chunk, so peak memory scales with spatial chunk size regardless of
        how `time` itself is chunked. Defaults to None, which uses `gfetch.mosaic.
        load`'s own default of `{"x": 64, "y": 64}`.
    compute_chunk_factor : int | None
        Number of `chunks` per dask chunk (the `mosaic` stage's processing brick)
        along each of `x` and `y`. Must divide `shard_factor`. Larger bricks mean
        fewer, larger dask tasks: more memory in use, less scheduling overhead.
        Memory per brick in flight is roughly brick side² x time steps x bands x 2
        bytes, since the median holds the brick's whole time axis. Defaults to None,
        which uses `gfetch.mosaic.resolve_compute_chunks`' default of 16 (1024 px
        bricks at the default `chunks`).
    shard_factor : int | None
        Number of `chunks` per Zarr shard of the `mosaic` output, along each of `x`
        and `y` (e.g. `32` means 32x32 chunks per shard). A shard is one file holding
        several chunks, and the `mosaic` stage's unit of work: each is computed and
        written by one task, skipped if already written, and recomputed as a whole
        otherwise. Its output (all bands) is held in memory at once. `1` disables
        sharding (one file per chunk). Reading a sharded store with GDAL (and QGIS)
        needs GDAL 3.13+. Defaults to None, which uses
        `gfetch.mosaic.resolve_shards`' default of 128.
    """

    time_range: TimeRangeConfig
    output_dir: Path
    aoi: AOIConfig | None = None
    countries: list[str] | None = None
    satellite: str = "sentinel-2"
    source: str | None = None
    bands: list[str] | None = None
    max_cloud_cover: float | None = None
    orbit_state: str | None = None
    collection: str | None = None
    cloud_mask_band: str | None = None
    cloud_mask_out: list[int] = field(default_factory=list)
    ocm: bool = False
    ocm_model_dir: Path | None = None
    resolution: float = 10.0
    n_workers: int = 4
    resampling: dict[str, str] = field(default_factory=dict)
    n_compute_workers: Any = None
    chunks: dict[str, int] | None = None
    compute_chunk_factor: int | None = None
    shard_factor: int | None = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir).expanduser().absolute()
        if self.ocm_model_dir is not None:
            self.ocm_model_dir = Path(self.ocm_model_dir).expanduser().absolute()

    @property
    def resolved_aoi(self) -> AOIConfig:
        """
        Returns
        -------
        AOIConfig
            `aoi`, resolved by `load()` (from `countries`, if `aoi` wasn't given
            directly) - never None on a `Config` returned by `load()`. `aoi` itself
            stays `AOIConfig | None` at the schema level so OmegaConf accepts a
            config that gives `countries` instead.
        """
        assert self.aoi is not None, "aoi not yet resolved - build this Config via load()"
        return self.aoi

    @cached_property
    def aoi_geometry(self) -> BaseGeometry:
        """
        Returns
        -------
        BaseGeometry
            The exact AOI shape in EPSG:4326, see `resolve_aoi_geometry`.
        """
        return resolve_aoi_geometry(self.resolved_aoi, self.countries)

    @property
    def cache_dir(self) -> Path:
        """
        Returns
        -------
        Path
            Directory holding the downloaded asset cache.
        """
        return self.output_dir / "cache"

    @property
    def ocm_model_path(self) -> Path:
        """
        Returns
        -------
        Path
            `ocm_model_dir` if set, else `<output_dir>/ocm_models`.
        """
        return (
            self.ocm_model_dir if self.ocm_model_dir is not None else self.output_dir / "ocm_models"
        )

    @property
    def items_path(self) -> Path:
        """
        Returns
        -------
        Path
            Hand-off file written by the `search` stage: the raw (remote-href)
            search results, read back by the `download` stage.
        """
        return self.output_dir / "items.json"

    @property
    def cached_items_path(self) -> Path:
        """
        Returns
        -------
        Path
            Hand-off file written by the `download` stage: the same items with
            asset hrefs rewritten to the local cache, read back by the `mosaic`
            stage.
        """
        return self.output_dir / "cached_items.json"

    @property
    def coverage_path(self) -> Path:
        """
        Returns
        -------
        Path
            GeoParquet written by the `coverage` stage: the item counts per mosaic
            shard.
        """
        return self.output_dir / "coverage.parquet"

    def zarr_path(self, crs: CRS) -> Path:
        """
        Output Zarr store path for one UTM zone's mosaic, written by the `write`
        stage. The AOI may span several UTM zones, each written to its own store -
        see `gfetch.mosaic.group_by_utm_zone`.

        Parameters
        ----------
        crs : CRS
            UTM zone this mosaic was computed in.

        Returns
        -------
        Path
            Output Zarr store path for this zone.
        """
        return self.output_dir / f"mosaic_epsg{crs.epsg}.zarr"


def load(path: Path, satellite_key: str) -> Config:
    """
    Load and validate one satellite's job configuration from a unified YAML file.

    The file's top level holds generic defaults; only the ones that are also
    `Config` fields apply here (a generic field that's only meaningful elsewhere,
    e.g. GEDI's `fields`, is ignored, not an error). The reserved section keys
    `s1`/`s2`/`gedi_l2a`/`gedi_l4a`/`custom` override those defaults for that
    satellite only (`gedi_*` are irrelevant here - `gfetch.cli.gedi_config.load`
    reads them), and the selected section *is* validated strictly (an unknown key
    there raises). A built-in (`s1`/`s2`) section's `satellite` is force-set from
    `BUILTIN_SATELLITES` regardless of what the section itself contains; a
    `custom.<name>` section must supply `collection` and `bands` explicitly (there's
    no `gfetch.profiles`/`gfetch.sources` registry entry for an arbitrary satellite
    name), and defaults `satellite` to `<name>` if not given. Either way, `output_dir` defaults to
    `<generic output_dir>/<satellite_key>` unless the section sets its own.

    Parameters
    ----------
    path : Path
        Path to the configuration YAML file.
    satellite_key : str
        Which satellite to load: `'s1'`, `'s2'`, or a name under the file's
        `custom:` section.

    Returns
    -------
    Config
        Fully validated configuration object.

    Raises
    ------
    ValueError
        If `satellite_key` is neither a built-in nor a key under `custom:`, a
        `custom` section is missing `collection`/`bands`, neither/both of
        `aoi`/`countries` are given (see `resolve_aoi`), or `ocm` is set with a
        config it can't apply to (see `_validate_ocm`).
    """
    log.debug(f"Loading config from {path} (satellite={satellite_key!r})")
    raw = OmegaConf.load(path)
    assert isinstance(raw, DictConfig), f"{path} must be a YAML mapping, not a list"
    config_field_names = {f.name for f in fields(Config)}
    generic = OmegaConf.create(
        {k: v for k, v in raw.items() if k in config_field_names and k not in RESERVED_SECTION_KEYS}
    )

    is_builtin = satellite_key in BUILTIN_SATELLITES
    if is_builtin:
        section = raw.get(satellite_key, OmegaConf.create({}))
        assert isinstance(section, DictConfig)
    else:
        custom = raw.get("custom", OmegaConf.create({}))
        assert isinstance(custom, DictConfig)
        if satellite_key not in custom:
            raise ValueError(
                f"Unknown satellite {satellite_key!r} in {path}; known: "
                f"{sorted(BUILTIN_SATELLITES)}, custom: {sorted(str(k) for k in custom)}"
            )
        section = custom[satellite_key]
        assert isinstance(section, DictConfig)
        missing = [f for f in ("collection", "bands") if f not in section]
        if missing:
            raise ValueError(
                f"custom satellite {satellite_key!r} in {path} is missing required "
                f"field(s) {missing}"
            )

    overrides = OmegaConf.merge(generic, section)
    assert isinstance(overrides, DictConfig)
    if is_builtin:
        # Force-set regardless of what the section/generic dict contains.
        overrides["satellite"] = BUILTIN_SATELLITES[satellite_key]
    elif overrides.get("satellite") is None:
        overrides["satellite"] = satellite_key

    if "output_dir" not in section and "output_dir" in overrides:
        overrides["output_dir"] = str(Path(str(overrides["output_dir"])) / satellite_key)

    structured = OmegaConf.structured(Config)
    merged = OmegaConf.merge(structured, overrides)
    OmegaConf.resolve(merged)
    cfg: Config = OmegaConf.to_object(merged)  # type: ignore[assignment]
    context = f"{path} (satellite={satellite_key!r})"
    cfg.aoi = resolve_aoi(cfg.aoi, cfg.countries, context=context)
    _validate_ocm(cfg, context=context)
    log.debug(f"Resolved output_dir={cfg.output_dir}")
    return cfg


def _validate_ocm(cfg: Config, *, context: str) -> None:
    """
    Check that a config with `ocm` set can actually be masked with OmniCloudMask.

    Parameters
    ----------
    cfg : Config
        Job configuration.
    context : str
        Description of what's being validated (e.g. the config path), prefixed to
        the error message.

    Raises
    ------
    ValueError
        If `ocm` is set on a satellite other than Sentinel-2, together with
        `cloud_mask_band`, or without every one of `OCM_INPUT_BANDS` in the bands.
    """
    if not cfg.ocm:
        return
    if cfg.satellite != "sentinel-2":
        raise ValueError(f"{context}: `ocm` only applies to sentinel-2, set it under `s2:`")
    if cfg.cloud_mask_band is not None:
        raise ValueError(f"{context}: `ocm` and `cloud_mask_band` are mutually exclusive")
    missing = [band for band in OCM_INPUT_BANDS if band not in resolve_bands(cfg)]
    if missing:
        raise ValueError(f"{context}: `ocm` needs bands {missing}, add them to `bands`")


def resolve_bands(cfg: Config) -> list[str]:
    """
    Resolve the asset keys to download/load for a config.

    Parameters
    ----------
    cfg : Config
        Job configuration.

    Returns
    -------
    list[str]
        `cfg.bands` if set, else `cfg.satellite`'s `gfetch.profiles.SatelliteProfile.
        default_bands` if it has one, else an empty list (a `custom` satellite with
        no `bands` override and no profile).
    """
    if cfg.bands:
        return list(cfg.bands)
    try:
        return list(get_profile(cfg.satellite).default_bands)
    except ValueError:
        return []


def resolve_output_variables(cfg: Config) -> list[str]:
    """
    Resolve the data variables the `mosaic` stage writes for a config.

    Parameters
    ----------
    cfg : Config
        Job configuration.

    Returns
    -------
    list[str]
        `resolve_bands(cfg)`, or with `orbit_state: as_bands`, one
        `{band}_{orbit_state}` variable per band and orbit state (see
        `gfetch.mosaic.orbit_state_variables`).
    """
    bands = resolve_bands(cfg)
    if cfg.orbit_state == ORBIT_STATE_AS_BANDS:
        from gfetch.mosaic import orbit_state_variables

        return orbit_state_variables(bands)
    return bands


def resolve_source(cfg: Config) -> str:
    """
    Resolve the STAC source to search for a config.

    Parameters
    ----------
    cfg : Config
        Job configuration.

    Returns
    -------
    str
        `cfg.source` if set, else `cfg.satellite`'s `gfetch.profiles.
        SatelliteProfile.default_source` if it has a profile, else 'earthsearch'.
    """
    if cfg.source is not None:
        return cfg.source
    try:
        return get_profile(cfg.satellite).default_source
    except ValueError:
        return "earthsearch"


def resolve_cloud_mask(cfg: Config) -> tuple[str | None, frozenset[int]]:
    """
    Resolve the cloud-mask band/classification-values-to-exclude for a config.

    Parameters
    ----------
    cfg : Config
        Job configuration.

    Returns
    -------
    tuple[str | None, frozenset[int]]
        OmniCloudMask's `(OCM_BAND, OCM_MASK_OUT)` if `cfg.ocm` is set, else
        `(cfg.cloud_mask_band, cfg.cloud_mask_out)` if `cloud_mask_band` is set, else
        `cfg.satellite`'s `gfetch.profiles.SatelliteProfile` mask if it has one, else
        `(None, frozenset())` (no masking - e.g. a `custom` satellite with no
        override and no profile, or a profile with no cloud-mask band like
        Sentinel-1).
    """
    if cfg.ocm:
        return OCM_BAND, OCM_MASK_OUT
    if cfg.cloud_mask_band is not None:
        return cfg.cloud_mask_band, frozenset(cfg.cloud_mask_out)
    try:
        profile = get_profile(cfg.satellite)
        return profile.cloud_mask_band, profile.cloud_mask_out
    except ValueError:
        return None, frozenset()


def resolve_compute_workers(cfg: Config) -> tuple[int, int]:
    """
    Resolve a config's `n_compute_workers` into a worker-count range.

    Parameters
    ----------
    cfg : Config
        Job configuration.

    Returns
    -------
    tuple[int, int]
        `(min, max)` dask thread counts, equal when the count is fixed.

    Raises
    ------
    ValueError
        If `n_compute_workers` is neither None, a positive int, nor a `[min, max]`
        pair of positive ints with `min <= max`.
    """
    value = cfg.n_compute_workers
    if value is None:
        return available_cpus(), available_cpus()
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value, value
    if (
        isinstance(value, list | tuple)
        and len(value) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
        and 1 <= value[0] <= value[1]
    ):
        return value[0], value[1]
    raise ValueError(
        f"n_compute_workers must be a positive int or a [min, max] pair with "
        f"1 <= min <= max, got {value!r}"
    )
