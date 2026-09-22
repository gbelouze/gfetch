"""Main entry point for the gfetch CLI."""

import logging
from pathlib import Path

import cyclopts

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


@app.command
def search(config: Path, verbose: bool = False) -> None:
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

    search_cmd(config)


@app.command
def download(config: Path, verbose: bool = False) -> None:
    """
    Download the assets of a job's searched items into a local cache.

    Internet-connected stage; safe to run on an HPC login/data-transfer node. Safe to
    resume after being killed/preempted, and safe for multiple concurrent runs to
    share the same cache directory.

    Parameters
    ----------
    config : Path
        Path to the configuration YAML file.
    verbose : bool
        Enable verbose (DEBUG) logging. Defaults to False.
    """
    _setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    from gfetch.cli.download import download as download_cmd

    download_cmd(config)


@app.command
def mosaic(config: Path, task_id: int = 0, n_tasks: int = 1, verbose: bool = False) -> None:
    """
    Load, cloud-mask, composite, and write a job's items to a Zarr mosaic.

    Compute-only stage; no internet access required once `gfetch download` has been
    run (loads from remote hrefs otherwise). Safe to resume after being killed, and
    safe to split across several concurrent invocations via `task_id`/`n_tasks` (e.g.
    a SLURM job array), each writing disjoint patches of the same output store.

    Parameters
    ----------
    config : Path
        Path to the configuration YAML file.
    task_id : int
        This invocation's index among `n_tasks` concurrent invocations. Defaults to 0.
    n_tasks : int
        Total number of concurrent invocations splitting this job's patches between
        them. Defaults to 1 (no splitting).
    verbose : bool
        Enable verbose (DEBUG) logging. Defaults to False.
    """
    _setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    from gfetch.cli.mosaic import mosaic as mosaic_cmd

    mosaic_cmd(config, task_id=task_id, n_tasks=n_tasks)


@app.command
def gedi(config: Path, verbose: bool = False) -> None:
    """
    Fetch GEDI L2A footprints matching a configuration's AOI/time range via
    SlideRule and write them to GeoParquet.

    Standalone command, independent of the raster search/download/mosaic pipeline
    and its `Config` schema: SlideRule resolves matching granules and subsets them
    server-side, so there's no separate search/download stage and no local asset
    cache. Uses its own config schema, `gfetch.cli.gedi_config.GediConfig`.

    Parameters
    ----------
    config : Path
        Path to the configuration YAML file.
    verbose : bool
        Enable verbose (DEBUG) logging. Defaults to False.
    """
    _setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    from gfetch.cli.gedi import gedi as gedi_cmd

    gedi_cmd(config)


if __name__ == "__main__":
    app()
