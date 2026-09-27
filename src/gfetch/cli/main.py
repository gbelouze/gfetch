"""Main entry point for the gfetch CLI."""

import logging
from pathlib import Path

import cyclopts

from gfetch.cli.config import BUILTIN_SATELLITES

app = cyclopts.App(
    name="gfetch",
    help="CLI for downloading analysis-ready Earth observation datasets from STAC.",
)


def _setup_logging(level: int = logging.INFO) -> None:
    from gfetch.utils.log import setup

    setup(level=level)


@app.command
def init(output: Path | None = None, force: bool = False) -> None:
    """
    Write a gfetch configuration template.

    Parameters
    ----------
    output : Path | None
        Path to save the configuration template. Defaults to None, which saves to
        './config.yaml'.
    force : bool
        Overwrite an existing configuration file. Defaults to False.
    """
    _setup_logging()
    from gfetch.cli.init import init as init_cmd

    init_cmd(output, force)


def _register_raster_commands(sub_app: cyclopts.App, satellite_key: str) -> None:
    """
    Register `search`/`download`/`mosaic` on a built-in satellite's sub-app.

    Parameters
    ----------
    sub_app : cyclopts.App
        Sub-app to register commands on (e.g. `gfetch s2`).
    satellite_key : str
        Which satellite this sub-app loads from a job config - `'s1'` or `'s2'`.
        See `gfetch.cli.config.load`.
    """

    @sub_app.command(name="search")
    def _search(config: Path, verbose: bool = False) -> None:
        """
        Search a STAC source for items matching a configuration's AOI/time range.

        Internet-connected stage; safe to run on an HPC login/data-transfer node.

        Parameters
        ----------
        config : Path
            Path to the configuration YAML file.
        verbose : bool
            Enable verbose (DEBUG) logging. Defaults to False.
        """
        _setup_logging(level=logging.DEBUG if verbose else logging.INFO)
        from gfetch.cli.search import search as search_cmd

        search_cmd(config, satellite_key)

    @sub_app.command(name="download")
    def _download(config: Path, verbose: bool = False) -> None:
        """
        Download the assets of a job's searched items into a local cache.

        Internet-connected stage; safe to run on an HPC login/data-transfer node.
        Safe to resume after being killed/preempted, and safe for multiple
        concurrent runs to share the same cache directory.

        Parameters
        ----------
        config : Path
            Path to the configuration YAML file.
        verbose : bool
            Enable verbose (DEBUG) logging. Defaults to False.
        """
        _setup_logging(level=logging.DEBUG if verbose else logging.INFO)
        from gfetch.cli.download import download as download_cmd

        download_cmd(config, satellite_key)

    @sub_app.command(name="mosaic")
    def _mosaic(config: Path, task_id: int = 0, n_tasks: int = 1, verbose: bool = False) -> None:
        """
        Load, cloud-mask, composite, and write a job's items to a Zarr mosaic.

        Compute-only stage; no internet access required once `download` has been
        run (loads from remote hrefs otherwise). Safe to resume after being killed,
        and safe to split across several concurrent invocations via
        `task_id`/`n_tasks` (e.g. a SLURM job array), each writing disjoint shards
        of the same output store.

        Parameters
        ----------
        config : Path
            Path to the configuration YAML file.
        task_id : int
            This invocation's index among `n_tasks` concurrent invocations.
            Defaults to 0.
        n_tasks : int
            Total number of concurrent invocations splitting this job's shards
            between them. Defaults to 1 (no splitting).
        verbose : bool
            Enable verbose (DEBUG) logging. Defaults to False.
        """
        _setup_logging(level=logging.DEBUG if verbose else logging.INFO)
        from gfetch.cli.mosaic import mosaic as mosaic_cmd

        mosaic_cmd(config, satellite_key, task_id=task_id, n_tasks=n_tasks)

    @sub_app.command(name="vrt")
    def _vrt(config: Path, verbose: bool = False) -> None:
        """
        Write a multi-band VRT next to each of a job's Zarr mosaics.

        Opens in QGIS (or anything GDAL-based) as one raster with a band per mosaic
        variable, where the Zarr driver alone shows one band at a time. Copies no
        pixel data. Reading a sharded mosaic needs GDAL >= 3.13. Safe to run before
        `mosaic` has finished (unwritten shards read as nodata), and to rerun.

        Parameters
        ----------
        config : Path
            Path to the configuration YAML file.
        verbose : bool
            Enable verbose (DEBUG) logging. Defaults to False.
        """
        _setup_logging(level=logging.DEBUG if verbose else logging.INFO)
        from gfetch.cli.finalize import vrt as vrt_cmd

        vrt_cmd(config, satellite_key)

    @sub_app.command(name="clean")
    def _clean(config: Path, verbose: bool = False) -> None:
        """
        Delete a job's download cache once every one of its Zarr mosaics is complete.

        Refuses, deleting nothing, if any mosaic is incomplete. Safe to rerun.

        Parameters
        ----------
        config : Path
            Path to the configuration YAML file.
        verbose : bool
            Enable verbose (DEBUG) logging. Defaults to False.
        """
        _setup_logging(level=logging.DEBUG if verbose else logging.INFO)
        from gfetch.cli.finalize import clean as clean_cmd

        clean_cmd(config, satellite_key)


for _satellite_key, _satellite_name in BUILTIN_SATELLITES.items():
    _sub_app = cyclopts.App(
        name=_satellite_key,
        help=f"{_satellite_name} via the raster search/download/mosaic pipeline.",
    )
    _register_raster_commands(_sub_app, _satellite_key)
    app.command(_sub_app)


custom_app = cyclopts.App(
    name="custom",
    help=(
        "Arbitrary STAC satellite/source defined under a job config's `custom:` "
        "section (no gfetch.profiles/gfetch.sources registry entry needed)."
    ),
)


@custom_app.command(name="search")
def _custom_search(name: str, config: Path, verbose: bool = False) -> None:
    """
    Search a STAC source for items matching a `custom` satellite's AOI/time range.

    Internet-connected stage; safe to run on an HPC login/data-transfer node.

    Parameters
    ----------
    name : str
        Which entry under `config`'s `custom:` section to load.
    config : Path
        Path to the configuration YAML file.
    verbose : bool
        Enable verbose (DEBUG) logging. Defaults to False.
    """
    _setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    from gfetch.cli.search import search as search_cmd

    search_cmd(config, name)


@custom_app.command(name="download")
def _custom_download(name: str, config: Path, verbose: bool = False) -> None:
    """
    Download the assets of a `custom` satellite job's searched items into a local
    cache.

    Internet-connected stage; safe to run on an HPC login/data-transfer node. Safe
    to resume after being killed/preempted, and safe for multiple concurrent runs
    to share the same cache directory.

    Parameters
    ----------
    name : str
        Which entry under `config`'s `custom:` section to load.
    config : Path
        Path to the configuration YAML file.
    verbose : bool
        Enable verbose (DEBUG) logging. Defaults to False.
    """
    _setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    from gfetch.cli.download import download as download_cmd

    download_cmd(config, name)


@custom_app.command(name="mosaic")
def _custom_mosaic(
    name: str, config: Path, task_id: int = 0, n_tasks: int = 1, verbose: bool = False
) -> None:
    """
    Load, cloud-mask, composite, and write a `custom` satellite job's items to a
    Zarr mosaic.

    Compute-only stage; no internet access required once `download` has been run
    (loads from remote hrefs otherwise). Safe to resume after being killed, and
    safe to split across several concurrent invocations via `task_id`/`n_tasks`
    (e.g. a SLURM job array), each writing disjoint shards of the same output
    store.

    Parameters
    ----------
    name : str
        Which entry under `config`'s `custom:` section to load.
    config : Path
        Path to the configuration YAML file.
    task_id : int
        This invocation's index among `n_tasks` concurrent invocations. Defaults
        to 0.
    n_tasks : int
        Total number of concurrent invocations splitting this job's shards
        between them. Defaults to 1 (no splitting).
    verbose : bool
        Enable verbose (DEBUG) logging. Defaults to False.
    """
    _setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    from gfetch.cli.mosaic import mosaic as mosaic_cmd

    mosaic_cmd(config, name, task_id=task_id, n_tasks=n_tasks)


@custom_app.command(name="vrt")
def _custom_vrt(name: str, config: Path, verbose: bool = False) -> None:
    """
    Write a multi-band VRT next to each of a `custom` satellite job's Zarr mosaics.

    Opens in QGIS (or anything GDAL-based) as one raster with a band per mosaic
    variable, where the Zarr driver alone shows one band at a time. Copies no pixel
    data. Reading a sharded mosaic needs GDAL >= 3.13. Safe to run before `mosaic`
    has finished (unwritten shards read as nodata), and to rerun.

    Parameters
    ----------
    name : str
        Which entry under `config`'s `custom:` section to load.
    config : Path
        Path to the configuration YAML file.
    verbose : bool
        Enable verbose (DEBUG) logging. Defaults to False.
    """
    _setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    from gfetch.cli.finalize import vrt as vrt_cmd

    vrt_cmd(config, name)


@custom_app.command(name="clean")
def _custom_clean(name: str, config: Path, verbose: bool = False) -> None:
    """
    Delete a `custom` satellite job's download cache once every one of its Zarr
    mosaics is complete.

    Refuses, deleting nothing, if any mosaic is incomplete. Safe to rerun.

    Parameters
    ----------
    name : str
        Which entry under `config`'s `custom:` section to load.
    config : Path
        Path to the configuration YAML file.
    verbose : bool
        Enable verbose (DEBUG) logging. Defaults to False.
    """
    _setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    from gfetch.cli.finalize import clean as clean_cmd

    clean_cmd(config, name)


app.command(custom_app)


gedi_app = cyclopts.App(
    name="gedi",
    help=(
        "GEDI footprints via SlideRule, written to GeoParquet. Standalone, independent "
        "of the raster search/download/mosaic pipeline: SlideRule resolves matching "
        "granules and subsets them server-side, so there's no separate search/download "
        "stage and no local asset cache."
    ),
)


@gedi_app.command(name="l2a")
def _gedi_l2a(config: Path, verbose: bool = False) -> None:
    """
    Fetch GEDI L2A (elevation, relative height) footprints matching a
    configuration's AOI/time range, from its `gedi_l2a:` section.

    Parameters
    ----------
    config : Path
        Path to the configuration YAML file.
    verbose : bool
        Enable verbose (DEBUG) logging. Defaults to False.
    """
    _setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    from gfetch.cli.gedi import gedi as gedi_cmd

    gedi_cmd(config, "l2a")


@gedi_app.command(name="l4a")
def _gedi_l4a(config: Path, verbose: bool = False) -> None:
    """
    Fetch GEDI L4A (aboveground biomass density) footprints matching a
    configuration's AOI/time range, from its `gedi_l4a:` section.

    Parameters
    ----------
    config : Path
        Path to the configuration YAML file.
    verbose : bool
        Enable verbose (DEBUG) logging. Defaults to False.
    """
    _setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    from gfetch.cli.gedi import gedi as gedi_cmd

    gedi_cmd(config, "l4a")


app.command(gedi_app)


if __name__ == "__main__":
    app()
