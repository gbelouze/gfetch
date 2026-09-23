import logging
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


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
    default_path = Path("config.yaml")
    output_path = output if output is not None else default_path
    output_path = output_path.expanduser().resolve().absolute()

    if output_path.exists() and not force:
        log.error(f"{output_path} already exists. Use --force to overwrite.")
        return

    if output_path.suffix == "":
        output_path.mkdir()
        output_path = output_path / "config.yaml"

    config_dict = {
        "aoi": {"left": 2.2, "bottom": 48.7, "right": 2.5, "top": 49.0},
        "time_range": {"start": "2024-01-01", "end": "2024-06-01"},
        "output_dir": str(output_path.parent),
        # Per-satellite sections override the generic fields above for their own
        # `gfetch <key> <verb>` run - see `gfetch.cli.config.load`. `s1`/`gedi`/
        # `custom` sections are also available; add whichever you need.
        "s2": {"source": "earthsearch", "bands": ["red", "green", "blue"]},
    }

    with output_path.open("w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    log.info("Configuration template created!")
    log.info(f"  Path: {output_path.absolute()}")
    log.info("  Edit the file to customize your dataset configuration.")
