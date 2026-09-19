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
def mosaic(config: Path, verbose: bool = False) -> None:
    """
    Load, cloud-mask, composite, and write a job's items to a Zarr mosaic.

    Compute-only stage; no internet access required once `gfetch download` has been
    run (loads from remote hrefs otherwise).

    Parameters
    ----------
    config : Path
        Path to the configuration YAML file.
    verbose : bool
        Enable verbose (DEBUG) logging. Defaults to False.
    """
    _setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    from gfetch.cli.mosaic import mosaic as mosaic_cmd

    mosaic_cmd(config)


if __name__ == "__main__":
    app()
