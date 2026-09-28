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
import time
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import pystac
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling

from gfetch.profiles import OCM_BAND, OCM_INPUT_BANDS, OCM_NODATA

log = logging.getLogger(__name__)

__all__ = [
    "OCM_RESOLUTION",
    "OcmInput",
    "fetch_models",
    "load_models",
    "ocm_grid",
    "ocm_path",
    "predict",
    "read_input",
    "resolve_device",
    "set_num_threads",
    "with_ocm_asset",
    "write_mask",
    "write_ocms",
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


def resolve_device(device: str | None) -> str:
    """
    Pick the torch device inference runs on.

    Parameters
    ----------
    device : str | None
        Torch device, e.g. 'cuda' or 'cpu'. Defaults to None, which uses a GPU if
        one is available.

    Returns
    -------
    str
        `device` if given, else 'cuda' when available, else 'cpu'.
    """
    if device is not None:
        return device
    _omnicloudmask()
    import torch  # pyrefly: ignore[missing-import]

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_models(model_dir: Path, *, device: str, dtype: str) -> list[Any]:
    """
    Load OmniCloudMask's model ensemble onto `device`, once for a whole run.

    Parameters
    ----------
    model_dir : Path
        Directory holding the weights, see `fetch_models`. Weights missing from it
        are downloaded, which needs internet access.
    device : str
        Torch device, see `resolve_device`.
    dtype : str
        Inference dtype, e.g. 'float32' or 'float16'.

    Returns
    -------
    list[Any]
        The ensemble's torch modules, to pass to `predict`.
    """
    _omnicloudmask()
    import torch  # pyrefly: ignore[missing-import]
    from omnicloudmask.cloud_mask import collect_models  # pyrefly: ignore[missing-import]

    if device.startswith("cuda"):
        # Every batch has the same patch shape, so cuDNN's per-shape autotuning pays off.
        torch.backends.cudnn.benchmark = True
    return collect_models(
        custom_models=None,
        inference_device=torch.device(device),
        inference_dtype=getattr(torch, dtype),
        source="hugging_face",
        destination_model_dir=model_dir,
    )


def predict(
    bands: np.ndarray,
    *,
    models: list[Any],
    device: str,
    dtype: str,
    batch_size: int,
) -> np.ndarray:
    """
    Predict an OmniCloudMask cloud/shadow mask from red, green and NIR reflectances.

    Parameters
    ----------
    bands : np.ndarray
        Red, green and NIR bands stacked as `(3, height, width)`, with 0 as no-data.
    models : list[Any]
        Model ensemble, see `load_models`.
    device : str
        Torch device the models live on.
    dtype : str
        Inference dtype the models were loaded with.
    batch_size : int
        Number of patches per forward pass.

    Returns
    -------
    np.ndarray
        `uint8` classes, `(height, width)`: 0 clear, 1 thick cloud, 2 thin cloud,
        3 cloud shadow, `OCM_NODATA` wherever any input band is 0.
    """
    ocm = _omnicloudmask()
    classes = ocm.predict_from_array(
        bands.astype(np.float32),
        custom_models=models,
        inference_device=device,
        inference_dtype=dtype,
        batch_size=batch_size,
    )[0].astype(np.uint8)
    classes[(bands == 0).any(axis=0)] = OCM_NODATA
    return classes


@dataclass(frozen=True)
class OcmInput:
    """An item's `OCM_INPUT_BANDS`, read at `OCM_RESOLUTION`."""

    bands: np.ndarray
    transform: Affine
    crs: CRS


def read_input(item: pystac.Item) -> OcmInput:
    """
    Read a cached item's `OCM_INPUT_BANDS`, averaged down to `OCM_RESOLUTION`.

    Parameters
    ----------
    item : pystac.Item
        Cached item whose `OCM_INPUT_BANDS` assets point at local files.

    Returns
    -------
    OcmInput
        The stacked bands and their grid.
    """
    hrefs = [item.assets[band].get_absolute_href() for band in OCM_INPUT_BANDS]
    with rasterio.open(hrefs[0]) as src:
        shape, transform = ocm_grid((src.height, src.width), src.transform)
        crs = src.crs
    bands = []
    for href in hrefs:
        with rasterio.open(href) as src:
            bands.append(src.read(1, out_shape=shape, resampling=Resampling.average))
    return OcmInput(np.stack(bands), transform, crs)


def write_mask(path: Path, classes: np.ndarray, transform: Affine, crs: CRS) -> None:
    """
    Write a mask to `path`, atomically.

    Parameters
    ----------
    path : Path
        Output GeoTIFF, see `ocm_path`.
    classes : np.ndarray
        Mask, see `predict`.
    transform : Affine
        The mask's affine transform.
    crs : CRS
        The mask's CRS.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".tif")
    os.close(fd)
    try:
        with rasterio.open(
            tmp,
            "w",
            driver="GTiff",
            height=classes.shape[0],
            width=classes.shape[1],
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


def write_ocms(
    jobs: Iterable[tuple[pystac.Item, Path]],
    *,
    models: list[Any],
    device: str,
    dtype: str,
    batch_size: int,
    prefetch: int = 2,
) -> Iterator[pystac.Item]:
    """
    Compute and write each item's mask, overlapping I/O with inference.

    Up to `prefetch` items are read in background threads while the current one is
    on the GPU, and each mask is written in a background thread too: reading and
    writing GeoTIFFs mostly runs in GDAL, which releases the GIL.

    Parameters
    ----------
    jobs : Iterable[tuple[pystac.Item, Path]]
        Cached items, each with its output path (see `ocm_path`).
    models : list[Any]
        Model ensemble, see `load_models`.
    device : str
        Torch device the models live on.
    dtype : str
        Inference dtype the models were loaded with.
    batch_size : int
        Number of patches per forward pass.
    prefetch : int
        Number of items read ahead. Defaults to 2.

    Yields
    ------
    pystac.Item
        Each item, once its mask is written.
    """
    jobs = iter(jobs)
    with (
        ThreadPoolExecutor(max_workers=prefetch) as readers,
        ThreadPoolExecutor(max_workers=1) as writer,
    ):
        reads: deque[tuple[pystac.Item, Path, Future[OcmInput]]] = deque(
            (item, path, readers.submit(read_input, item)) for item, path in islice(jobs, prefetch)
        )
        written: tuple[pystac.Item, Future[None]] | None = None
        while reads:
            item, path, read = reads.popleft()
            for next_item, next_path in islice(jobs, 1):
                reads.append((next_item, next_path, readers.submit(read_input, next_item)))
            start = time.perf_counter()
            data = read.result()
            waited = time.perf_counter() - start
            classes = predict(
                data.bands, models=models, device=device, dtype=dtype, batch_size=batch_size
            )
            log.debug(
                f"{item.id}: waited {waited:.1f}s on reading, "
                f"predicted in {time.perf_counter() - start - waited:.1f}s"
            )
            if written is not None:
                written[1].result()
                yield written[0]
            written = (item, writer.submit(write_mask, path, classes, data.transform, data.crs))
        if written is not None:
            written[1].result()
            yield written[0]
