"""Plot whatever `results/*.csv` files exist into a single grid figure - one
panel per benchmark run found, skipping any that haven't been run yet. Reads
CSVs only; never runs a benchmark itself.

Depends on `matplotlib` (a real dev dependency) and `pandas` (pulled in
transitively via `xarray`).

No dependency on the gfetch package itself - standalone, notebook-style.
"""

# %%
import logging
from collections.abc import Iterable
from pathlib import Path

import cyclopts
import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec
import matplotlib.pyplot as plt
import pandas as pd
from common import setup_logging

log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "plots.png"

# Fixed categorical order (dataviz skill's validated default palette, light
# mode) - colors are assigned by position in this list, never auto-cycled, so
# the same label gets the same color across panels.
PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
SCENARIO_COLOR = {"wide": PALETTE[0], "deep": PALETTE[1]}

app = cyclopts.App()


def _read_csv_safe(path: Path) -> pd.DataFrame | None:
    """Read `path` as CSV, tolerating a run still writing to it.

    A benchmark still in progress may have just created a results file
    (`.open("w", ...)` truncates it immediately) with its header not yet
    flushed to disk, or a partially-written last row - both are transient,
    not corruption, so this logs a warning and returns None instead of
    raising.

    Parameters
    ----------
    path : Path
        CSV file to read.

    Returns
    -------
    pd.DataFrame | None
        The parsed CSV, or None if it couldn't be read yet.
    """
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        log.warning(f"{path}: empty (benchmark likely still running) - skipping")
        return None
    except pd.errors.ParserError as e:
        log.warning(f"{path}: unparseable, likely mid-write - skipping ({e})")
        return None


def _color_map(labels: Iterable) -> dict:
    """Assign each of `labels` (deduped, sorted) a fixed `PALETTE` slot.

    Parameters
    ----------
    labels : Iterable
        Values to assign colors to (e.g. distinct `num_workers` in one plot).

    Returns
    -------
    dict
        `{label: hex color}`, in ascending-sorted order of `labels`.
    """
    uniq = sorted(set(labels))
    return {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(uniq)}


def _legend_unique(ax: plt.Axes) -> None:
    """Draw `ax`'s legend with duplicate labels collapsed to one entry each.

    `_mark_timed_out()` adds a `"timed out (inferred)"` entry per line/group
    plotted on `ax`, which would otherwise repeat once per group.

    Parameters
    ----------
    ax : plt.Axes
        Axes whose legend to draw.
    """
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=True))
    ax.legend(by_label.values(), by_label.keys())


def _mark_timed_out(ax: plt.Axes, df: pd.DataFrame, x: str, y: str) -> None:
    """Overlay an "x" marker on any `timed_out` rows - inferred, not measured.

    Parameters
    ----------
    ax : plt.Axes
        Axes already holding the normal line/bar for `df`.
    df : pd.DataFrame
        Rows plotted on `ax`, including a `timed_out` column.
    x : str
        Column plotted on the x-axis.
    y : str
        Column plotted on the y-axis.
    """
    if "timed_out" not in df.columns:
        return
    timed_out = df[df["timed_out"]]
    if not timed_out.empty:
        ax.scatter(
            timed_out[x],
            timed_out[y],
            marker="x",
            color="black",
            s=50,
            zorder=5,
            label="timed out (inferred)",
        )


def plot_sweep_by_scenario(
    ax: plt.Axes, results_dir: Path, glob_pattern: str, x: str, y: str, title: str
) -> bool:
    """Plot `y` vs `x`, one line per `<name>_<scenario>.csv` file found.

    Parameters
    ----------
    ax : plt.Axes
        Axes to draw into.
    results_dir : Path
        Directory to glob for result CSVs in.
    glob_pattern : str
        `results_dir`-relative glob, e.g. `"concurrency_*.csv"`.
    x : str
        Column to sweep on the x-axis.
    y : str
        Column to plot on the y-axis (typically `"mb_s"`).
    title : str
        Axes title.

    Returns
    -------
    bool
        True if any matching file was found and plotted.
    """
    paths = sorted(results_dir.glob(glob_pattern))
    if not paths:
        return False

    plotted = False
    for path in paths:
        df = _read_csv_safe(path)
        if df is None or df.empty:
            continue
        scenario = path.stem.rsplit("_", 1)[-1]
        df = df.sort_values(x)
        color = SCENARIO_COLOR.get(scenario, "gray")
        ax.plot(df[x], df[y], marker="o", color=color, label=scenario)
        _mark_timed_out(ax, df, x, y)
        plotted = True

    if not plotted:
        return False
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    _legend_unique(ax)
    return True


def plot_two_stage_download_concurrency(ax: plt.Axes, results_dir: Path) -> bool:
    """Plot `two_stage_download_concurrency.csv`: `max_concurrency` vs `mb_s` per scenario.

    Parameters
    ----------
    ax : plt.Axes
        Axes to draw into.
    results_dir : Path
        Directory to look for the CSV in.

    Returns
    -------
    bool
        True if the file was found and plotted.
    """
    path = results_dir / "two_stage_download_concurrency.csv"
    df = _read_csv_safe(path) if path.exists() else None
    if df is None or df.empty:
        return False

    for scenario in sorted(df["scenario"].unique()):
        group = df[df["scenario"] == scenario].sort_values("max_concurrency")
        color = SCENARIO_COLOR.get(scenario, "gray")
        ax.plot(group["max_concurrency"], group["mb_s"], marker="o", color=color, label=scenario)

    ax.set_title("two-stage: download concurrency sweep")
    ax.set_xlabel("max_concurrency")
    ax.set_ylabel("mb_s")
    ax.legend()
    return True


def plot_download_full(ax: plt.Axes, results_dir: Path, filename: str, title: str) -> bool:
    """Plot a `<scenario, n_items, n_bytes, elapsed_s, mb_s>` CSV: one bar per scenario.

    Parameters
    ----------
    ax : plt.Axes
        Axes to draw into.
    results_dir : Path
        Directory to look for the CSV in.
    filename : str
        `results_dir`-relative CSV filename, e.g. `"two_stage_download_full.csv"`.
    title : str
        Axes title.

    Returns
    -------
    bool
        True if the file was found and plotted.
    """
    path = results_dir / filename
    df = _read_csv_safe(path) if path.exists() else None
    if df is None or df.empty:
        return False

    colors = [SCENARIO_COLOR.get(s, "gray") for s in df["scenario"]]
    ax.bar(df["scenario"], df["mb_s"], color=colors)
    ax.set_title(title)
    ax.set_xlabel("scenario")
    ax.set_ylabel("mb_s")
    return True


def plot_load_sweep(
    fig: plt.Figure,
    gridspec: matplotlib.gridspec.SubplotSpec,
    results_dir: Path,
    filename: str,
    title_prefix: str,
) -> bool:
    """Plot a `<scenario, chunk, num_workers, ..., mb_s>` CSV: one panel per scenario.

    Parameters
    ----------
    fig : plt.Figure
        Figure to add panels to.
    gridspec : matplotlib.gridspec.SubplotSpec
        One grid cell to split into one sub-panel per scenario found.
    results_dir : Path
        Directory to look for the CSV in.
    filename : str
        `results_dir`-relative CSV filename, e.g. `"two_stage_load.csv"`.
    title_prefix : str
        Prefix for each sub-panel's title, before `" [<scenario>]"`.

    Returns
    -------
    bool
        True if the file was found and plotted.
    """
    path = results_dir / filename
    df = _read_csv_safe(path) if path.exists() else None
    if df is None or df.empty:
        return False

    scenarios = sorted(df["scenario"].unique())
    worker_colors = _color_map(df["num_workers"].unique())
    inner = gridspec.subgridspec(1, len(scenarios))

    for i, scenario in enumerate(scenarios):
        ax = fig.add_subplot(inner[0, i])
        scenario_df = df[df["scenario"] == scenario]
        for num_workers in sorted(scenario_df["num_workers"].unique()):
            group = scenario_df[scenario_df["num_workers"] == num_workers].sort_values("chunk")
            ax.plot(
                group["chunk"],
                group["mb_s"],
                marker="o",
                color=worker_colors[num_workers],
                label=f"workers={num_workers}",
            )
            _mark_timed_out(ax, group, "chunk", "mb_s")
        ax.set_title(f"{title_prefix} [{scenario}]")
        ax.set_xlabel("chunk")
        ax.set_ylabel("mb_s")
        _legend_unique(ax)
    return True


def plot_gdal_config(ax: plt.Axes, scenario: str, path: Path) -> bool:
    """Plot one `gdal_config_<scenario>.csv`: grouped bars, config x num_workers.

    Parameters
    ----------
    ax : plt.Axes
        Axes to draw into.
    scenario : str
        Scenario name, for the panel title.
    path : Path
        CSV to read.

    Returns
    -------
    bool
        True if the file was found, non-empty, and plotted.
    """
    df = _read_csv_safe(path)
    if df is None or df.empty:
        return False

    configs = sorted(df["config"].unique())
    worker_colors = _color_map(df["num_workers"].unique())
    width = 0.8 / max(len(worker_colors), 1)

    for i, num_workers in enumerate(sorted(worker_colors)):
        group = df[df["num_workers"] == num_workers].set_index("config").reindex(configs)
        offsets = [j + i * width for j in range(len(configs))]
        ax.bar(
            offsets,
            group["mb_s"],
            width=width,
            color=worker_colors[num_workers],
            label=f"workers={num_workers}",
        )

    ax.set_xticks([j + 0.4 for j in range(len(configs))])
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_title(f"gdal_config [{scenario}]")
    ax.set_ylabel("mb_s")
    ax.legend()
    return True


@app.default
def main(
    results_dir: Path = RESULTS_DIR, output: Path = DEFAULT_OUTPUT, verbose: bool = False
) -> None:
    """Plot every available `results/*.csv` into one grid figure.

    Parameters
    ----------
    results_dir : Path
        Directory to look for result CSVs in. Defaults to `RESULTS_DIR`.
    output : Path
        PNG path to save the figure to. Defaults to `DEFAULT_OUTPUT`.
    verbose : bool
        Log at DEBUG instead of INFO. Defaults to False.
    """
    setup_logging(verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    panels = [
        (
            "concurrency",
            lambda ax: plot_sweep_by_scenario(
                ax, results_dir, "concurrency_*.csv", "num_workers", "mb_s", "concurrency sweep"
            ),
        ),
        (
            "chunks",
            lambda ax: plot_sweep_by_scenario(
                ax, results_dir, "chunks_*.csv", "chunk", "mb_s", "chunk-size sweep"
            ),
        ),
        (
            "download_concurrency",
            lambda ax: plot_two_stage_download_concurrency(ax, results_dir),
        ),
        (
            "two_stage_download_full",
            lambda ax: plot_download_full(
                ax, results_dir, "two_stage_download_full.csv", "two-stage: full download"
            ),
        ),
        (
            "gfetch_pipeline_download",
            lambda ax: plot_download_full(
                ax,
                results_dir,
                "gfetch_pipeline_download.csv",
                "gfetch pipeline: download",
            ),
        ),
    ]

    def _make_gdal_config_panel(path: Path, scenario: str):
        return lambda ax: plot_gdal_config(ax, scenario, path)

    for path in sorted(results_dir.glob("gdal_config_*.csv")):
        scenario = path.stem.rsplit("_", 1)[-1]
        panels.append((f"gdal_config_{scenario}", _make_gdal_config_panel(path, scenario)))

    # These span a full grid cell each, split internally into one sub-panel per
    # scenario found - plotted separately from `panels` since they need a
    # `SubplotSpec` (to subgrid), not a plain `Axes`.
    load_sweep_panels = [
        ("two_stage_load", "two_stage_load.csv", "two-stage load"),
        ("gfetch_pipeline_load", "gfetch_pipeline_load.csv", "gfetch pipeline load"),
    ]
    load_sweep_panels = [
        (name, filename, title)
        for name, filename, title in load_sweep_panels
        if (results_dir / filename).exists()
    ]

    n_slots = len(panels) + len(load_sweep_panels)
    if n_slots == 0:
        log.warning(f"no result CSVs found in {results_dir}")
        return

    ncols = 2
    nrows = (n_slots + ncols - 1) // ncols
    fig = plt.figure(figsize=(6 * ncols, 4 * nrows))
    gridspecs = fig.add_gridspec(nrows, ncols)

    used = 0
    for i, (name, plot_fn) in enumerate(panels):
        ax = fig.add_subplot(gridspecs[i // ncols, i % ncols])
        if plot_fn(ax):
            log.info(f"plotted {name}")
            used += 1
        else:
            fig.delaxes(ax)

    for j, (name, filename, title) in enumerate(load_sweep_panels):
        i = len(panels) + j
        if plot_load_sweep(fig, gridspecs[i // ncols, i % ncols], results_dir, filename, title):
            log.info(f"plotted {name}")
            used += 1

    if used == 0:
        log.warning(f"no result CSVs found in {results_dir}")
        plt.close(fig)
        return

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    log.info(f"saved {used} panel(s) to {output}")


# %%
if __name__ == "__main__":
    app()
