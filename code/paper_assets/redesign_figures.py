from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_assets.paths import FIGURES, REPO_DATA, REPO_EXPERIMENT4_RESULTS

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GREY = "#6B7280"



def use_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            # The figures this module ships are drawn at their printed
            # width, so these are the sizes the reader sees.
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "axes.titleweight": "regular",
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "legend.fontsize": 8,
            "legend.frameon": False,
            # Type 3 is matplotlib's default and is refused by most
            # typesetters; 42 embeds TrueType outlines instead
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "grid.color": "#CBD5E1",
            "grid.linewidth": 0.42,
            "grid.alpha": 0.42,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            # anything rasterized goes out at print resolution
            "savefig.dpi": 600,
            "axes.formatter.use_mathtext": True,
        }
    )


#: Width of the manuscript's text block, in inches.  ``\the\textwidth``
#: is 384 pt and a TeX point is 1/72.27 in.
TEXT_WIDTH = 384.0 / 72.27


def polish_axis(axis: mpl.axes.Axes, *, grid: bool = True) -> None:
    axis.tick_params(which="both", top=False, right=False)
    if grid:
        axis.grid(True, which="major")
        axis.grid(False, which="minor")


def save_figure(
    figure: mpl.figure.Figure,
    output: Path,
    stem: str,
    *,
    png: bool = True,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output / f"{stem}.pdf",
        dpi=300,
        metadata={"Creator": "RFM figure redesign"},
    )
    if png:
        figure.savefig(output / f"{stem}.png", dpi=300)
    plt.close(figure)


def _decimate(x: np.ndarray, y: np.ndarray, maximum: int = 600) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= maximum:
        return x, y
    index = np.unique(np.linspace(0, x.size - 1, maximum).astype(int))
    return x[index], y[index]


def plot_appendix_convergence(output: Path) -> None:
    """The baseline histories, as two figures rather than one crowded panel row.

    Five panels across one text width left every axis too small to read.  The
    eigenvalue trajectories and the reactor errors are separate subjects and are
    referred to separately in the appendix, so they are drawn separately.
    """
    eigen_figure, eigen_axes = plt.subplots(
        1, 2, figsize=(0.92 * TEXT_WIDTH, 2.15)
    )
    reactor_figure, reactor_axes = plt.subplots(
        1, 2, figsize=(0.92 * TEXT_WIDTH, 2.35)
    )
    axes = (*eigen_axes, *reactor_axes)
    for axis in axes:
        axis.tick_params(labelsize=7.5)
        axis.xaxis.label.set_size(8.5)
        axis.yaxis.label.set_size(8.5)
        axis.title.set_size(9)

    cube_progress = pd.read_csv(REPO_DATA / "ji2024_drm_validated_progress.csv")
    cube_terminal = pd.read_csv(REPO_DATA / "ji2024_drm_validated_terminal.csv")
    mode_colors = (BLUE, ORANGE, GREEN)
    titles = {"square": r"(a) 10D DRM: $V(t)=t^2$", "exp": r"(b) 10D DRM: $V(t)=e^{-\pi t}$"}
    for axis, kind in zip(axes[0:2], ("square", "exp"), strict=True):
        for color, mode in zip(mode_colors, (1, 2, 3), strict=True):
            rows = cube_progress.loc[(cube_progress["kind"] == kind) & (cube_progress["mode"] == mode)].sort_values("epoch")
            terminal = cube_terminal.loc[(cube_terminal["kind"] == kind) & (cube_terminal["mode"] == mode)].iloc[0]
            selected_index = (rows["epoch"] - int(terminal["selected_epoch"])).abs().idxmin()
            selected = rows.loc[selected_index]
            x, y = _decimate(rows["epoch"].to_numpy(), rows["relerr_eps"].to_numpy())
            axis.loglog(x, y, color=color, linewidth=0.85, alpha=0.78, label=rf"$\lambda_{mode}$")
            axis.plot(selected["epoch"], selected["relerr_eps"], marker="o", color=color, markersize=3.8, linestyle="none")
            stopped = rows.iloc[-1]
            axis.plot(stopped["epoch"], stopped["relerr_eps"], marker="s", markerfacecolor="white", markeredgecolor=color, markersize=3.8, linestyle="none")
        axis.axvline(1.0e5, color=GREY, linestyle=(0, (4, 2)), linewidth=0.8)
        axis.set_title(titles[kind], loc="left")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Relative eigenvalue error")
        axis.set_xlim(8.0e2, 2.4e5)
        axis.set_ylim(1.0e-4, 2.0e1)
        axis.legend(fontsize=7, ncol=3, loc="upper right", columnspacing=0.65, handlelength=1.2)

    reference = np.load(REPO_EXPERIMENT4_RESULTS / "reference_subdiv48.npz")
    reference_keff = float(reference["keff"])
    reactor_methods = (
        ("drm_seed2026071101", "DRM", ORANGE),
        ("gipmnn_seed2026071102", "GIPMNN", GREEN),
        ("pc_gipmnn_seed2026071103", "PC-GIPMNN", PURPLE),
    )
    for directory, label, color in reactor_methods:
        rows = pd.read_csv(REPO_EXPERIMENT4_RESULTS / directory / "convergence.csv")
        epochs = rows["epoch"].to_numpy()
        keff_error = np.abs(rows["keff"].to_numpy() - reference_keff) / abs(reference_keff)
        flux_error = rows["flux_error"].to_numpy()
        summary = json.loads((REPO_EXPERIMENT4_RESULTS / directory / "summary.json").read_text(encoding="utf-8"))
        selected_index = int(np.argmin(np.abs(epochs - int(summary["selected_epoch"]))))
        for axis, values in zip(axes[2:4], (keff_error, flux_error), strict=True):
            x, y = _decimate(epochs, values)
            axis.loglog(x, y, color=color, linewidth=0.9, alpha=0.82, label=label)
            axis.plot(epochs[selected_index], values[selected_index], marker="o", color=color, markersize=3.8, linestyle="none")
            axis.plot(epochs[-1], values[-1], marker="s", markerfacecolor="white", markeredgecolor=color, markersize=3.8, linestyle="none")
    for axis in axes[2:4]:
        axis.axvline(5.0e4, color=GREY, linestyle=(0, (4, 2)), linewidth=0.8)
        axis.axvline(5.0e5, color="#9CA3AF", linestyle=(0, (2, 2)), linewidth=0.8)
        axis.set_xlabel("Epoch")
    axes[2].set_title(r"(a) IAEA: $k_{\rm eff}$", loc="left")
    axes[2].set_ylabel(r"Relative error in $k_{\rm eff}$")
    axes[2].legend(fontsize=7, ncol=1, loc="upper right", handlelength=1.3)
    axes[3].set_title("(b) IAEA: scalar flux", loc="left")
    axes[3].set_ylabel("Relative max error")

    for axis in axes:
        polish_axis(axis)
    eigen_figure.tight_layout()
    reactor_figure.tight_layout()
    save_figure(eigen_figure, output, "appendix_baseline_convergence_eigenvalue")
    save_figure(reactor_figure, output, "appendix_baseline_convergence_reactor")


def build_all(output: Path) -> None:
    use_publication_style()
    plot_appendix_convergence(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redraw the appendix baseline-convergence figures "
                                     "without touching scientific data.")
    parser.add_argument(
        "--output",
        type=Path,
        default=FIGURES,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_all(args.output.resolve())


if __name__ == "__main__":
    main()
