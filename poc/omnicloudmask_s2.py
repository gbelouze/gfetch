# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cyclopts",
#     "matplotlib",
#     "numpy",
#     "omnicloudmask>=1.7",
#     "pystac-client",
#     "rasterio",
#     "rich",
# ]
# ///
"""POC: compute an OmniCloudMask (OCM) cloud/shadow mask on a cached Sentinel-2 L2A item
and compare it with the L2A Scene Classification (SCL) mask gfetch uses today.

OCM needs red, green and NIR. A gfetch cache item only holds the bands that were
requested (red/green/blue/scl by default), so red/green are read from the cache and
the matching NIR window is read remotely from the item's Earth Search COG (only the
window's tiles are fetched). Its output classes are 0 clear, 1 thick cloud, 2 thin
cloud, 3 cloud shadow.

Standalone, no dependency on the gfetch package:

    uv run poc/omnicloudmask_s2.py --help
    uv run poc/omnicloudmask_s2.py ~/Data/gfetch/s2/cache/S2B_T36MYA_20240719T080518_L2A
"""

import logging
import time
from pathlib import Path

import cyclopts
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from omnicloudmask import predict_from_array  # pyrefly: ignore[missing-import]
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.windows import Window
from rich.logging import RichHandler

log = logging.getLogger(__name__)

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
DEFAULT_ITEM_DIR = Path.home() / "Data/gfetch/s2/cache/S2B_T36MYA_20240719T080518_L2A"
NATIVE_RES = 10
SCL_RES = 20
# Same set as gfetch's `_SENTINEL2_SCL_MASK_OUT`, minus 0 (no-data), which is kept
# apart so it doesn't count as cloud in the comparison.
SCL_CLOUD = (1, 3, 7, 8, 9, 10)
SCL_SHADOW = 3
OCM_LABELS = ("clear", "thick cloud", "thin cloud", "cloud shadow")
OCM_COLORS = ("#00000000", "#ffd400", "#00b4ff", "#ff2d8a")

app = cyclopts.App(help=__doc__)


def cloudiest_window(scl_path: Path, size: int, overview_factor: int = 20) -> Window:
    """
    Find the `size`x`size` window (10 m pixels) with the most SCL cloud and no no-data.

    Parameters
    ----------
    scl_path : Path
        Path to the item's SCL GeoTIFF (20 m).
    size : int
        Window side length in 10 m pixels.
    overview_factor : int
        Decimation factor (on the 10 m grid) used to scan the SCL cheaply.
        Defaults to 20.

    Returns
    -------
    Window
        Window on the 10 m grid.
    """
    with rasterio.open(scl_path) as src:
        shape = (src.height * SCL_RES // (NATIVE_RES * overview_factor),) * 2
        scl = src.read(1, out_shape=shape, resampling=Resampling.nearest)
    cloud = np.isin(scl, SCL_CLOUD).astype(float)
    nodata = (scl == 0).astype(float)
    k = size // overview_factor
    step = max(k // 4, 1)
    best, best_score = (0, 0), -1.0
    for row in range(0, shape[0] - k + 1, step):
        for col in range(0, shape[1] - k + 1, step):
            if nodata[row : row + k, col : col + k].mean() > 0.01:
                continue
            score = cloud[row : row + k, col : col + k].mean()
            if score > best_score:
                best, best_score = (row, col), score
    if best_score < 0:
        raise ValueError(f"No {size}px window without no-data in {scl_path}")
    log.info(f"Picked window with {best_score:.1%} SCL cloud/shadow")
    return Window(best[1] * overview_factor, best[0] * overview_factor, size, size)


def read_band(path: str | Path, window: Window, out_size: int, res: int) -> np.ndarray:
    """
    Read a single-band raster over a 10 m-grid window, resampled to `out_size`.

    Parameters
    ----------
    path : str | Path
        Local path or remote URL of the band.
    window : Window
        Window on the 10 m grid.
    out_size : int
        Output side length in pixels.
    res : int
        Native resolution of the band, in meters.

    Returns
    -------
    np.ndarray
        The resampled band, of shape `(out_size, out_size)`.
    """
    scale = NATIVE_RES / res
    band_window = Window(
        window.col_off * scale,
        window.row_off * scale,
        window.width * scale,
        window.height * scale,
    )
    resampling = Resampling.nearest if res == SCL_RES else Resampling.average
    with rasterio.open(path) as src:
        return src.read(
            1, window=band_window, out_shape=(out_size, out_size), resampling=resampling
        )


def remote_nir_href(item_id: str) -> str:
    """
    Look up the NIR (B08) COG href of a Sentinel-2 item on Earth Search.

    Parameters
    ----------
    item_id : str
        Earth Search item id, i.e. the cache item directory name.

    Returns
    -------
    str
        HTTPS href of the item's `nir` asset.
    """
    item = next(Client.open(EARTH_SEARCH_URL).search(ids=[item_id]).items())
    return item.assets["nir"].href


def plot(
    rgb: np.ndarray,
    scl: np.ndarray,
    ocm: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """
    Save a side-by-side RGB / SCL mask / OCM mask figure.

    Parameters
    ----------
    rgb : np.ndarray
        Raw reflectance, shape `(3, H, W)`, in red/green/blue order.
    scl : np.ndarray
        SCL classes, shape `(H, W)`.
    ocm : np.ndarray
        OCM classes, shape `(H, W)`.
    title : str
        Figure title.
    out_path : Path
        Output PNG path.
    """
    valid = rgb[0] > 0
    lo, hi = np.percentile(rgb[:, valid], (2, 98))
    display = np.clip((rgb.transpose(1, 2, 0) - lo) / (hi - lo), 0, 1)
    scl_classes = np.where(scl == SCL_SHADOW, 3, np.where(np.isin(scl, SCL_CLOUD), 1, 0))
    cmap = ListedColormap(OCM_COLORS)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7.5), sharex=True, sharey=True)
    axes[0].imshow(display)
    axes[0].set_title("RGB")
    for ax, mask, name in (
        (axes[1], scl_classes, "SCL (cloud-ish classes / shadow)"),
        (axes[2], ocm, "OmniCloudMask"),
    ):
        ax.imshow(display)
        ax.imshow(mask, cmap=cmap, vmin=0, vmax=3, alpha=0.6, interpolation="nearest")
        ax.set_title(name)
    for ax in axes:
        ax.set_axis_off()
    fig.legend(
        handles=[
            Patch(color=c, label=label)
            for c, label in zip(OCM_COLORS[1:], OCM_LABELS[1:], strict=True)
        ],
        loc="lower center",
        ncol=3,
    )
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


@app.default
def main(
    item_dir: Path = DEFAULT_ITEM_DIR,
    out_dir: Path = Path("poc/out"),
    window_size: int = 4000,
    resolution: int = 20,
    device: str | None = None,
) -> None:
    """
    Run OmniCloudMask on a window of a cached Sentinel-2 L2A item.

    Parameters
    ----------
    item_dir : Path
        gfetch cache directory of one item, holding `red.tif`, `green.tif` and
        `scl.tif`. Defaults to a cached T36MYA scene with ~17% cloud cover.
    out_dir : Path
        Where to write the mask GeoTIFF and comparison PNG. Defaults to `poc/out`.
    window_size : int
        Side of the processed window, in 10 m pixels. The cloudiest SCL window of
        that size is picked. Defaults to 4000 (40 km).
    resolution : int
        Resolution OCM runs at, in meters; OCM is trained for 10-50 m. Defaults to 20.
    device : str | None
        Torch device, e.g. 'cpu', 'mps', 'cuda'. Defaults to OCM's own pick.
    """
    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()], format="%(message)s")
    out_dir.mkdir(parents=True, exist_ok=True)
    item_id = item_dir.name
    out_size = window_size * NATIVE_RES // resolution

    window = cloudiest_window(item_dir / "scl.tif", window_size)
    log.info(f"{item_id}: window {window}, running at {resolution} m ({out_size}px)")

    nir_href = remote_nir_href(item_id)
    t0 = time.perf_counter()
    red, green, blue = (
        read_band(item_dir / f"{band}.tif", window, out_size, NATIVE_RES)
        for band in ("red", "green", "blue")
    )
    nir = read_band(nir_href, window, out_size, NATIVE_RES)
    scl = read_band(item_dir / "scl.tif", window, out_size, SCL_RES)
    log.info(f"Read bands in {time.perf_counter() - t0:.1f}s (NIR from {nir_href})")

    t0 = time.perf_counter()
    ocm = predict_from_array(
        np.stack([red, green, nir]).astype(np.float32), inference_device=device
    )[0]
    log.info(f"OCM inference in {time.perf_counter() - t0:.1f}s")

    valid = scl != 0
    ocm_cloud = (ocm > 0) & valid
    scl_cloud = np.isin(scl, SCL_CLOUD) & valid
    n = valid.sum()
    for label, value in zip(OCM_LABELS, range(4), strict=True):
        log.info(f"OCM {label:>12}: {(ocm[valid] == value).mean():6.1%}")
    log.info(f"SCL masked-out     : {scl_cloud.sum() / n:6.1%}")
    log.info(f"both masked        : {(ocm_cloud & scl_cloud).sum() / n:6.1%}")
    log.info(f"OCM only           : {(ocm_cloud & ~scl_cloud).sum() / n:6.1%}")
    log.info(f"SCL only           : {(~ocm_cloud & scl_cloud).sum() / n:6.1%}")

    with rasterio.open(item_dir / "red.tif") as src:
        transform = src.window_transform(window) * src.transform.scale(resolution / NATIVE_RES)
        crs = src.crs
    mask_path = out_dir / f"{item_id}_ocm.tif"
    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        width=out_size,
        height=out_size,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(ocm.astype(np.uint8), 1)
    png_path = out_dir / f"{item_id}_ocm.png"
    plot(
        np.stack([red, green, blue]),
        scl,
        ocm,
        f"{item_id} @ {resolution} m",
        png_path,
    )
    log.info(f"Wrote {mask_path} and {png_path}")


if __name__ == "__main__":
    app()
