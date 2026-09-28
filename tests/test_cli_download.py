from pathlib import Path

import pystac
import yaml

from gfetch.cli.config import load
from gfetch.cli.download import download as download_cmd


def test_download_with_ocm_fetches_weights_and_no_mask_asset(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.dump(
            {
                "aoi": {"left": 2.2, "bottom": 48.7, "right": 2.21, "top": 48.71},
                "time_range": {"start": "2024-01-01", "end": "2024-06-01"},
                "output_dir": str(tmp_path),
                "s2": {"ocm": True, "bands": ["red", "green", "nir"]},
            }
        )
    )
    cfg = load(cfg_path, "s2")
    cfg.output_dir.mkdir(parents=True)
    pystac.ItemCollection([]).save_object(str(cfg.items_path))
    requested: list[list[str]] = []
    fetched: list[Path] = []

    async def fake_download_items(items, cache_dir, asset_keys, **kwargs):
        requested.append(list(asset_keys))
        return []

    monkeypatch.setattr("gfetch.cli.download.download_items", fake_download_items)
    monkeypatch.setattr("gfetch.ocm.fetch_models", fetched.append)

    download_cmd(cfg_path, "s2")

    assert requested == [["red", "green", "nir"]]
    assert fetched == [cfg.ocm_model_path]
