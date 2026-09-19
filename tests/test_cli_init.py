from pathlib import Path

from gfetch.cli.config import load
from gfetch.cli.init import init


def test_init_writes_loadable_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    init(config_path)

    assert config_path.exists()
    cfg = load(config_path)
    assert cfg.aoi.bbox
    assert cfg.time_range.datetime
    assert cfg.satellite == "sentinel-2"


def test_init_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    init(config_path)
    original = config_path.read_text()

    init(config_path)  # no force: should log an error and leave the file untouched

    assert config_path.read_text() == original


def test_init_overwrites_with_force(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    init(config_path)
    config_path.write_text("corrupted: true")

    init(config_path, force=True)

    cfg = load(config_path)
    assert cfg.satellite == "sentinel-2"
