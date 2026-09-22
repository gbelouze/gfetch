import logging
from dataclasses import dataclass, field
from pathlib import Path

from odc.geo.crs import CRS
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

__all__ = ["AOIConfig", "Config", "TimeRangeConfig", "load"]


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
    aoi : AOIConfig
        Area of interest.
    time_range : TimeRangeConfig
        Time range to search.
    output_dir : Path
        Directory holding the search/download hand-off files, the downloaded asset
        cache, and the output Zarr store.
    satellite : str
        Satellite/profile name (e.g. 'sentinel-2'). Defaults to 'sentinel-2'.
    source : str
        STAC source name (e.g. 'earthsearch', 'planetary-computer'). Defaults to
        'earthsearch'.
    bands : list[str] | None
        Asset keys to download/load. Defaults to None, which uses the satellite
        profile's default bands.
    max_cloud_cover : float | None
        Maximum `eo:cloud_cover` percentage to search for. Defaults to None (no
        filter). Only meaningful for optical satellites (e.g. Sentinel-2).
    orbit_state : str | None
        Restrict the search to one `sat:orbit_state` ('ascending' or 'descending'),
        e.g. to avoid blending SAR backscatter from different look geometries into one
        composite. Defaults to None (no filter, both orbit states included).
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
    n_compute_workers : int | None
        Number of dask threads used to compute and write the `mosaic` stage's
        per-zone output. Defaults to None, which uses `gfetch.utils.system.
        available_cpus` (the CPUs actually reserved for this process, e.g. by a
        SLURM job's `--cpus-per-task`, not the whole node). Lower this to bound how
        many chunks compute concurrently.
    chunks : dict[str, int] | None
        Dask chunk sizes for the `mosaic` stage's load step, e.g. `{"x": 512, "y":
        512}`. Smaller spatial chunks bound the memory a single output chunk's
        `gfetch.mosaic.composite` reduction needs: a non-associative reduction (e.g.
        the default 'median') must gather its whole `time` axis into memory per
        spatial chunk, so peak memory scales with spatial chunk size regardless of
        how `time` itself is chunked. Defaults to None, which uses `gfetch.mosaic.
        load`'s own default of `{"x": 2048, "y": 2048}`.
    patch_chunks : int
        Size of the `mosaic` stage's resumable/parallelizable write unit, as a
        multiple of `chunks`' native chunk size in both `x` and `y` (e.g. `10` means
        each unit spans a 10x10 block of native chunks). A unit is skipped entirely
        if every native chunk it covers is already written to the output Zarr store,
        and recomputed as a whole otherwise - never partially. Defaults to 1 (one
        native chunk per unit).
    """

    aoi: AOIConfig
    time_range: TimeRangeConfig
    output_dir: Path
    satellite: str = "sentinel-2"
    source: str = "earthsearch"
    bands: list[str] | None = None
    max_cloud_cover: float | None = None
    orbit_state: str | None = None
    resolution: float = 10.0
    n_workers: int = 4
    resampling: dict[str, str] = field(default_factory=dict)
    n_compute_workers: int | None = None
    chunks: dict[str, int] | None = None
    patch_chunks: int = 1

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir).expanduser().absolute()

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


def load(path: Path) -> Config:
    """
    Load and validate a configuration from a YAML file.

    Parameters
    ----------
    path : Path
        Path to the configuration YAML file.

    Returns
    -------
    Config
        Fully validated configuration object.
    """
    log.debug(f"Loading config from {path}")
    from_yaml = OmegaConf.load(path)
    structured = OmegaConf.structured(Config)
    merged = OmegaConf.merge(structured, from_yaml)
    OmegaConf.resolve(merged)
    cfg: Config = OmegaConf.to_object(merged)  # type: ignore[assignment]
    log.debug(f"Resolved output_dir={cfg.output_dir}")
    return cfg
