"""OmniCloudMask stage: compute a cloud/shadow mask per cached Sentinel-2 item.

Each item's mask lands at ``<cache_dir>/<item_id>/ocm.tif``, written to a temp file
and renamed into place, so the file's existence alone marks it complete. The `mosaic`
stage then masks with it instead of SCL, see `with_ocm_asset`. Needs the `ocm` extra
(`omnicloudmask`, which pulls in torch); inference runs on a GPU when one is visible.
"""

import copy
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pystac
import rasterio
from affine import Affine
from rasterio.enums import Resampling

from gfetch.profiles import OCM_BAND, OCM_INPUT_BANDS, OCM_NODATA

log = logging.getLogger(__name__)

__all__ = [
    "OCM_RESOLUTION",
    "fetch_models",
    "ocm_grid",
    "ocm_path",
    "predict",
    "set_num_threads",
    "with_ocm_asset",
    "write_ocm",
]

# OmniCloudMask is trained on 10-50 m imagery; 20 m matches SCL's own grid and costs
# a quarter of the inference of 10 m.
OCM_RESOLUTION = 20


def _omnicloudmask() -> Any:
    try:
        import omnicloudmask  # pyrefly: ignore[missing-import]
    except ImportError as e:
        log.error("omnicloudmask is not installed, install gfetch with the `ocm` extra")
        raise ImportError("OmniCloudMask needs the `ocm` extra: `uv sync --extra ocm`") from e
    return omnicloudmask


def ocm_path(cache_dir: Path, item_id: str) -> Path:
    """
    Locate an item's OmniCloudMask mask in the download cache.

    Parameters
    ----------
    cache_dir : Path
        Root cache directory, as passed to `gfetch.download.download_items`.
    item_id : str
        STAC item id.

    Returns
    -------
    Path
        `cache_dir/<item_id>/ocm.tif`, which exists only once the mask is complete.
    """
    return cache_dir / item_id / f"{OCM_BAND}.tif"


def ocm_grid(shape: tuple[int, int], transform: Affine) -> tuple[tuple[int, int], Affine]:
    """
    Derive the `OCM_RESOLUTION` grid covering the same extent as a band's grid.

    Parameters
    ----------
    shape : tuple[int, int]
        The band's `(height, width)`.
    transform : Affine
        The band's affine transform.

    Returns
    -------
    tuple[tuple[int, int], Affine]
        The mask's `(height, width)` and affine transform.
    """
    height, width = shape
    factor = OCM_RESOLUTION / abs(transform.a)
    out_shape = (round(height / factor), round(width / factor))
    return out_shape, transform @ Affine.scale(width / out_shape[1], height / out_shape[0])


def with_ocm_asset(item: pystac.Item, path: Path) -> pystac.Item:
    """
    Copy an item, adding its OmniCloudMask mask as the `OCM_BAND` asset.

    The asset's grid is derived from the item's `red` asset's `proj:shape`/
    `proj:transform` when it has them, see `ocm_grid`.

    Parameters
    ----------
    item : pystac.Item
        Cached item, as written by the `download` stage.
    path : Path
        The item's mask, see `ocm_path`.

    Returns
    -------
    pystac.Item
        A copy of `item` with an `OCM_BAND` asset pointing at `path`.
    """
    extra_fields: dict[str, Any] = {
        "raster:bands": [
            {"nodata": OCM_NODATA, "data_type": "uint8", "spatial_resolution": OCM_RESOLUTION}
        ]
    }
    red = item.assets[OCM_INPUT_BANDS[0]].extra_fields if OCM_INPUT_BANDS[0] in item.assets else {}
    if "proj:shape" in red and "proj:transform" in red:
        shape, transform = ocm_grid(tuple(red["proj:shape"]), Affine(*red["proj:transform"][:6]))
        extra_fields["proj:shape"] = list(shape)
        extra_fields["proj:transform"] = list(transform)[:6]

    result = copy.deepcopy(item)
    result.add_asset(
        OCM_BAND,
        pystac.Asset(
            href=str(path),
            media_type=pystac.MediaType.GEOTIFF,
            title="OmniCloudMask cloud/shadow mask",
            roles=["data"],
            extra_fields=extra_fields,
        ),
    )
    return result


def fetch_models(model_dir: Path) -> None:
    """
    Download OmniCloudMask's model weights, unless already in `model_dir`.

    Parameters
    ----------
    model_dir : Path
        Directory holding the weights, later read back by `predict`.
    """
    _omnicloudmask()
    from omnicloudmask.download_models import get_models  # pyrefly: ignore[missing-import]

    model_dir.mkdir(parents=True, exist_ok=True)
    get_models(model_dir=model_dir)
    log.info(f"OmniCloudMask weights ready in {model_dir}")


def set_num_threads(n: int) -> None:
    """
    Set the number of CPU threads torch uses for inference.

    torch otherwise sizes its thread pool to the whole node, not the CPUs actually
    reserved for this process (e.g. by SLURM's `--cpus-per-task`).

    Parameters
    ----------
    n : int
        Number of threads.
    """
    _omnicloudmask()
    import torch  # pyrefly: ignore[missing-import]

    torch.set_num_threads(n)


def predict(
    red: np.ndarray,
    green: np.ndarray,
    nir: np.ndarray,
    *,
    model_dir: Path,
    device: str | None = None,
) -> np.ndarray:
    """
    Predict an OmniCloudMask cloud/shadow mask from red, green and NIR reflectances.

    Parameters
    ----------
    red : np.ndarray
        Red band, 2D, with 0 as no-data.
    green : np.ndarray
        Green band, same shape as `red`.
    nir : np.ndarray
        NIR band, same shape as `red`.
    model_dir : Path
        Directory holding the model weights, see `fetch_models`. Weights missing
        from it are downloaded, which needs internet access.
    device : str | None
        Torch device, e.g. 'cuda' or 'cpu'. Defaults to None, which uses a GPU if
        one is available.

    Returns
    -------
    np.ndarray
        `uint8` classes, same shape as `red`: 0 clear, 1 thick cloud, 2 thin cloud,
        3 cloud shadow, `OCM_NODATA` wherever any input band is 0.
    """
    ocm = _omnicloudmask()
    classes = ocm.predict_from_array(
        np.stack([red, green, nir]).astype(np.float32),
        inference_device=device,
        destination_model_dir=model_dir,
    )[0].astype(np.uint8)
    classes[(red == 0) | (green == 0) | (nir == 0)] = OCM_NODATA
    return classes


def write_ocm(item: pystac.Item, path: Path, *, model_dir: Path, device: str | None = None) -> None:
    """
    Compute a cached item's OmniCloudMask mask and write it to `path`, atomically.

    The mask is computed at `OCM_RESOLUTION`, from the item's `OCM_INPUT_BANDS`
    assets averaged down to that resolution.

    Parameters
    ----------
    item : pystac.Item
        Cached item whose `OCM_INPUT_BANDS` assets point at local files.
    path : Path
        Output GeoTIFF, see `ocm_path`.
    model_dir : Path
        Directory holding the model weights, see `fetch_models`.
    device : str | None
        Torch device, see `predict`. Defaults to None.
    """
    hrefs = [item.assets[band].get_absolute_href() for band in OCM_INPUT_BANDS]
    with rasterio.open(hrefs[0]) as src:
        shape, transform = ocm_grid((src.height, src.width), src.transform)
        crs = src.crs
    bands = []
    for href in hrefs:
        with rasterio.open(href) as src:
            bands.append(src.read(1, out_shape=shape, resampling=Resampling.average))
    classes = predict(*bands, model_dir=model_dir, device=device)

    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".tif")
    os.close(fd)
    try:
        with rasterio.open(
            tmp,
            "w",
            driver="GTiff",
            height=shape[0],
            width=shape[1],
            count=1,
            dtype="uint8",
            nodata=OCM_NODATA,
            crs=crs,
            transform=transform,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            compress="deflate",
        ) as dst:
            dst.write(classes, 1)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    log.debug(f"{item.id}: wrote {path}")
