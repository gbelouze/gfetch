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
    """Plot a `<[source,] scenario, n_items, n_bytes, elapsed_s, mb_s>` CSV: one bar
    per scenario, or per source/scenario pair if a `source` column is present.

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

    has_source = "source" in df.columns
    labels = (
        [f"{s}\n{sc}" for s, sc in zip(df["source"], df["scenario"], strict=True)]
        if has_source
        else df["scenario"]
    )
    colors = [SCENARIO_COLOR.get(sc, "gray") for sc in df["scenario"]]

    ax.bar(labels, df["mb_s"], color=colors)
    ax.set_title(title)
    ax.set_xlabel("source / scenario" if has_source else "scenario")
    ax.set_ylabel("mb_s")
    return True


def plot_load_sweep(
    fig: plt.Figure,
    gridspec: matplotlib.gridspec.SubplotSpec,
    results_dir: Path,
    filename: str,
    title_prefix: str,
) -> bool:
    """Plot a `<[source,] scenario, chunk, num_workers, ..., mb_s>` CSV: one panel
    per scenario, or per source/scenario pair if a `source` column is present.

    Parameters
    ----------
    fig : plt.Figure
        Figure to add panels to.
    gridspec : matplotlib.gridspec.SubplotSpec
        One grid cell to split into one sub-panel per group found.
    results_dir : Path
        Directory to look for the CSV in.
    filename : str
        `results_dir`-relative CSV filename, e.g. `"two_stage_load.csv"`.
    title_prefix : str
        Prefix for each sub-panel's title, before `" [<group>]"`.

    Returns
    -------
    bool
        True if the file was found and plotted.
    """
    path = results_dir / filename
    df = _read_csv_safe(path) if path.exists() else None
    if df is None or df.empty:
        return False

    has_source = "source" in df.columns
    groups = (
        [(s, sc) for s in sorted(df["source"].unique()) for sc in sorted(df["scenario"].unique())]
        if has_source
        else [(None, sc) for sc in sorted(df["scenario"].unique())]
    )
    worker_colors = _color_map(df["num_workers"].unique())
    # 2 rows once there are more than 2 groups (source x scenario) - a single row
    # of 4 narrow sub-panels doesn't leave enough width for titles/legend to render
    # without overlapping; a 2x2 grid trades some of that back for height, which
    # these simple line plots don't need as much of.
    inner_rows = 2 if len(groups) > 2 else 1
    inner_cols = -(-len(groups) // inner_rows)  # ceil division
    inner = gridspec.subgridspec(inner_rows, inner_cols, hspace=0.5)

    # `title_prefix` is shown once, centered above the whole sub-panel grid, instead
    # of repeated in every sub-panel's own title - with up to 4 sub-panels, repeating
    # it per-panel overlapped adjacent titles.
    bbox = gridspec.get_position(fig)
    fig.text((bbox.x0 + bbox.x1) / 2, bbox.y1 + 0.005, title_prefix, ha="center", fontweight="bold")

    for i, (source, scenario) in enumerate(groups):
        ax = fig.add_subplot(inner[i // inner_cols, i % inner_cols])
        group_df = df[df["scenario"] == scenario]
        if source is not None:
            group_df = group_df[group_df["source"] == source]
        for num_workers in sorted(group_df["num_workers"].unique()):
            group = group_df[group_df["num_workers"] == num_workers].sort_values("chunk")
            ax.plot(
                group["chunk"],
                group["mb_s"],
                marker="o",
                color=worker_colors[num_workers],
                label=f"workers={num_workers}",
            )
            _mark_timed_out(ax, group, "chunk", "mb_s")
        label = f"{source}/{scenario}" if source is not None else scenario
        ax.set_title(f"[{label}]", fontsize=9)
        ax.set_xlabel("chunk")
        ax.set_ylabel("mb_s")
        if i == 0:
            _legend_unique(ax)
    return True


def plot_simple_sweep(
    ax: plt.Axes, results_dir: Path, filename: str, x: str, y: str, title: str
) -> bool:
    """Plot `y` vs `x` from a single, scenario-less CSV as one line.

    Parameters
    ----------
    ax : plt.Axes
        Axes to draw into.
    results_dir : Path
        Directory to look for the CSV in.
    filename : str
        `results_dir`-relative CSV filename, e.g. `"gedi_spatial_sweep.csv"`.
    x : str
        Column to sweep on the x-axis.
    y : str
        Column to plot on the y-axis.
    title : str
        Axes title.

    Returns
    -------
    bool
        True if the file was found, non-empty, and plotted.
    """
    path = results_dir / filename
    df = _read_csv_safe(path) if path.exists() else None
    if df is None or df.empty:
        return False

    df = df.sort_values(x)
    ax.plot(df[x], df[y], marker="o", color=PALETTE[0])
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
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
        (
            "gedi_spatial_sweep",
            lambda ax: plot_simple_sweep(
                ax,
                results_dir,
                "gedi_spatial_sweep.csv",
                "area_km2",
                "elapsed_s",
                "GEDI L2A: spatial AOI sweep",
            ),
        ),
        (
            "gedi_temporal_sweep",
            lambda ax: plot_simple_sweep(
                ax,
                results_dir,
                "gedi_temporal_sweep.csv",
                "n_days",
                "elapsed_s",
                "GEDI L2A: temporal range sweep",
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
