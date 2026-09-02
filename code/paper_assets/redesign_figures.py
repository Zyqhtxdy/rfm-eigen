from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

from iaea_repro.geometry import DOMAIN_SIZE, build_collocation, cell_labels, labels_at
from iaea_repro.metrics import mean_power_normalize
from iaea_repro.paper_assets import _load_baseline_flux
from iaea_repro.reference import load_reference
from iaea_repro.rfm import evaluate_random_feature_field
from paper_assets.paths import DATA, EXPERIMENT4_RESULTS, ROOT

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GOLD = "#E69F00"
GREY = "#6B7280"
BLACK = "#202124"
LIGHT_GREY = "#E5E7EB"



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


def tex_sci(value: float, digits: int = 2) -> str:
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10.0**exponent
    return rf"{mantissa:.{digits}f}\times10^{{{exponent}}}"


def _annotation_box(axis: mpl.axes.Axes, text: str, location: tuple[float, float]) -> None:
    axis.text(
        location[0],
        location[1],
        text,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=BLACK,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": LIGHT_GREY,
            "linewidth": 0.55,
            "alpha": 0.93,
        },
    )


def plot_experiment1(output: Path) -> None:
    frame = pd.read_csv(DATA / "experiment1_theory_matched_summary.csv").sort_values("N")
    metadata = json.loads(
        (DATA / "experiment1_theory_matched_metadata.json").read_text(encoding="utf-8")
    )
    guides = json.loads((DATA / "experiment1_rate_guides.json").read_text(encoding="utf-8"))
    budgets = frame["N"].to_numpy(dtype=float)
    gap = frame["h1_gap_q90"].to_numpy()
    eig = frame["cluster_relative_error_q90"].to_numpy()
    gap_reference = float(guides["gap_constant"]) * budgets ** (-0.5)
    eig_reference = float(guides["eigenvalue_constant"]) * budgets ** (-1.0)

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.78))
    panels = (
        (
            axes[0],
            gap,
            gap_reference,
            r"$Q_{0.9}(\operatorname{gap}_{H^1})$",
            r"$C_HN^{-1/2}$",
            r"(a) $H^1$ eigenspace gap",
            float(metadata["gap_q90_slope"]),
            metadata["gap_q90_slope_bootstrap_95"],
            "o",
        ),
        (
            axes[1],
            eig,
            eig_reference,
            r"$Q_{0.9}(e_{\rm cl})$",
            r"$C_\lambda N^{-1}$",
            "(b) Cluster eigenvalue error",
            float(metadata["eigenvalue_q90_slope"]),
            metadata["eigenvalue_q90_slope_bootstrap_95"],
            "^",
        ),
    )
    for axis, values, reference, ylabel, ref_label, title, slope, interval, marker in panels:
        axis.loglog(
            budgets,
            values,
            color=BLUE,
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=1.0,
            linewidth=1.65,
            label="RFM 90% quantile",
            zorder=3,
        )
        axis.loglog(
            budgets,
            reference,
            color=BLACK,
            linestyle=(0, (4, 2.5)),
            linewidth=1.05,
            label=rf"reference {ref_label}",
            zorder=2,
        )
        axis.set_title(title, loc="left")
        axis.set_xlabel(r"Number of features $N$")
        axis.set_ylabel(ylabel)
        axis.set_xticks(budgets)
        axis.set_xticklabels([str(int(value)) for value in budgets])
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xlim(budgets[0] * 0.95, budgets[-1] * 1.05)
        polish_axis(axis)
        _annotation_box(
            axis,
            rf"slope ${slope:.3f}$" + "\n" + rf"95% CI $[{interval[0]:.3f},{interval[1]:.3f}]$",
            (0.045, 0.055),
        )
        axis.legend(loc="upper right", handlelength=2.4)
    figure.subplots_adjust(left=0.085, right=0.995, bottom=0.22, top=0.90, wspace=0.31)
    save_figure(figure, output, "experiment1_rewrite_overview")


def plot_experiment1_ablation(output: Path) -> None:
    frame = pd.read_csv(DATA / "experiment1_cumulative_ablation.csv").sort_values(
        "contamination"
    )
    contamination = frame["contamination"].to_numpy()
    failure_rows = frame.loc[frame["correct_ordered_cluster"] == 0, "contamination"]
    threshold = float(failure_rows.iloc[0])

    figure, axes = plt.subplots(
        1, 2, figsize=(0.96 * TEXT_WIDTH, 2.30)
    )
    axes[0].plot(
        contamination,
        frame["cumulative_best_approximation_h1"],
        color=ORANGE,
        marker="s",
        markerfacecolor="white",
        markeredgewidth=0.95,
        label=r"$\eta_{W_\tau}(F_\star)$",
    )
    axes[0].plot(
        contamination,
        frame["ordered_h1_gap"],
        color=BLUE,
        marker="^",
        linestyle=(0, (4, 2)),
        label=r"ordered $H^1$ gap",
    )
    axes[0].set_title("(a) Subspace error and ordered gap", loc="left")
    axes[0].set_xlabel(r"Lower-mode contamination $\tau$")
    axes[0].set_ylabel("Subspace error")
    axes[0].set_ylim(-0.035, 1.04)
    axes[0].legend(loc="lower right")
    _annotation_box(
        axes[0],
        "target-space error\n"
        rf"$\max_\tau\eta_{{W_\tau}}(E_\star)="
        rf"{tex_sci(float(frame['target_best_approximation_h1'].max()), digits=1)}$",
        (0.44, 0.28),
    )

    axes[1].plot(
        contamination,
        frame["lambda7_ratio"],
        color=BLUE,
        marker="o",
        markerfacecolor=BLUE,
        label=r"$\lambda_{7,W_\tau}/\lambda_\star$",
        zorder=3,
    )
    axes[1].plot(
        contamination,
        frame["lambda8_ratio"],
        color=ORANGE,
        marker="s",
        markerfacecolor="white",
        markeredgewidth=1.0,
        label=r"$\lambda_{8,W_\tau}/\lambda_\star$",
        zorder=4,
    )
    axes[1].axhline(
        1.0,
        color=BLACK,
        linestyle=(0, (4, 2.5)),
        linewidth=1.0,
        label="target value",
    )
    axes[1].set_title("(b) Ordered Ritz values", loc="left")
    axes[1].set_xlabel(r"Lower-mode contamination $\tau$")
    axes[1].set_ylabel("Normalized Ritz value")
    axes[1].legend(loc="upper right")

    for axis in axes:
        axis.axvspan(threshold, contamination.max(), color=ORANGE, alpha=0.055, zorder=0)
        axis.axvline(threshold, color=GREY, linestyle=":", linewidth=1.0, zorder=1)
        polish_axis(axis)
    figure.subplots_adjust(left=0.085, right=0.995, bottom=0.22, top=0.91, wspace=0.31)
    save_figure(figure, output, "experiment1_cumulative_ablation")


def _experiment4_fields() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    solution = np.load(EXPERIMENT4_RESULTS / "rfm_random_fourier" / "solution.npz")
    reference = load_reference(EXPERIMENT4_RESULTS / "reference_subdiv48.npz")
    coordinates = np.linspace(0.0, DOMAIN_SIZE, 171)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel()])
    active = labels_at(points) > 0
    approximation = np.full(points.shape[0], np.nan)
    exact = np.full(points.shape[0], np.nan)
    approximation[active] = evaluate_random_feature_field(
        points[active], solution["modes"], solution["coefficients"]
    )
    exact[active] = reference.flux_at(points[active])
    approximation[active] /= max(float(np.nanmax(np.abs(approximation))), 1.0e-30)
    exact[active] /= max(float(np.nanmax(np.abs(exact))), 1.0e-30)
    return xx, yy, exact.reshape(xx.shape), approximation.reshape(xx.shape)


def plot_experiment4_overview(output: Path) -> None:
    xx, yy, exact, approximation = _experiment4_fields()
    material_colors = ["#F8FAFC", "#4C78A8", "#F2CF5B", "#D55E00", "#009E73"]
    material_cmap = ListedColormap(material_colors)
    material_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], material_cmap.N)
    field_cmap = mpl.colormaps["viridis"].copy()
    field_cmap.set_bad("#F8FAFC")

    figure, axes = plt.subplots(1, 3, figsize=(7.25, 2.55), constrained_layout=True)
    axes[0].imshow(
        cell_labels(),
        origin="lower",
        extent=(0, 170, 0, 170),
        cmap=material_cmap,
        norm=material_norm,
        interpolation="nearest",
    )
    axes[0].set_title("(a) Material map", loc="left")
    handles = [
        Patch(facecolor=material_colors[index], edgecolor="none", label=str(index))
        for index in range(1, 5)
    ]
    axes[0].legend(
        handles=handles,
        title="Material",
        title_fontsize=9.8,
        loc="upper right",
        ncol=2,
        frameon=True,
        framealpha=0.88,
        edgecolor=LIGHT_GREY,
        borderpad=0.25,
        handlelength=1.0,
        columnspacing=0.65,
    )
    field = axes[1].pcolormesh(
        xx, yy, exact, shading="auto", cmap=field_cmap, vmin=0.0, vmax=1.0, rasterized=True
    )
    axes[1].set_title("(b) Refined FV reference", loc="left")
    axes[2].pcolormesh(
        xx,
        yy,
        approximation,
        shading="auto",
        cmap=field_cmap,
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )
    axes[2].set_title("(c) RFM", loc="left")
    for index, axis in enumerate(axes):
        axis.set_aspect("equal")
        axis.set_xlim(0.0, DOMAIN_SIZE)
        axis.set_ylim(0.0, DOMAIN_SIZE)
        axis.set_xlabel(r"$x$ (cm)")
        axis.grid(False)
        if index == 0:
            axis.set_ylabel(r"$y$ (cm)")
        else:
            axis.set_yticklabels([])
    colorbar = figure.colorbar(field, ax=axes[1:], fraction=0.028, pad=0.018)
    colorbar.set_label("Normalized scalar flux", fontsize=11.0)
    colorbar.ax.tick_params(labelsize=9.6)
    save_figure(figure, output, "experiment4_rfm_overview")


def _load_experiment4_error_fields() -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, float],
]:
    solution = np.load(EXPERIMENT4_RESULTS / "rfm_random_fourier" / "solution.npz")
    reference = load_reference(EXPERIMENT4_RESULTS / "reference_subdiv48.npz")
    collocation = build_collocation(nx=171, points_per_cell=10)
    evaluation_points = collocation.interior
    evaluation_subdomains = collocation.subdomain
    reference_flux = mean_power_normalize(reference.flux_at(evaluation_points))
    fields: dict[str, np.ndarray] = {}
    acceptance = json.loads(
        (EXPERIMENT4_RESULTS / "baseline_acceptance.json").read_text(encoding="utf-8")
    )
    for record in acceptance:
        method = str(record["method"])
        seed = int(record["seed"])
        checkpoint = (
            EXPERIMENT4_RESULTS / "checkpoints" / f"{method}_seed{seed}_selected.pt"
        )
        fields[method] = mean_power_normalize(
            _load_baseline_flux(checkpoint, evaluation_points, evaluation_subdomains)
        )
    fields["rfm"] = mean_power_normalize(
        evaluate_random_feature_field(
            evaluation_points, solution["modes"], solution["coefficients"]
        )
    )
    reference_max = max(float(np.max(np.abs(reference_flux))), 1.0e-30)
    errors = {
        method: np.abs(values - reference_flux) / reference_max
        for method, values in fields.items()
    }
    maxima = {method: float(np.max(values)) for method, values in errors.items()}
    coordinates = np.linspace(0.0, DOMAIN_SIZE, 171)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    return xx, yy, errors, maxima


def plot_experiment4_errors(output: Path) -> None:
    xx, yy, errors, maxima = _load_experiment4_error_fields()
    points = np.column_stack([xx.ravel(), yy.ravel()])
    active = labels_at(points) > 0
    active_shape = active.reshape(xx.shape)
    positive = np.concatenate([values[values > 0.0] for values in errors.values()])
    vmax = max(maxima.values())
    vmin = max(float(np.quantile(positive, 0.002)), vmax * 1.0e-5)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = mpl.colormaps["magma"].copy()
    cmap.set_bad("#F8FAFC")
    labels = {"drm": "DRM", "gipmnn": "GIPMNN", "pc_gipmnn": "PC-GIPMNN", "rfm": "RFM"}
    methods = ("drm", "gipmnn", "pc_gipmnn", "rfm")

    figure, axes = plt.subplots(1, 4, figsize=(7.3, 2.42), constrained_layout=True)
    plot = None
    for panel, axis, method in zip("abcd", axes, methods, strict=True):
        grid = np.full(points.shape[0], np.nan)
        grid[active] = errors[method]
        plot = axis.pcolormesh(
            xx,
            yy,
            grid.reshape(xx.shape),
            shading="auto",
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
        axis.contour(
            xx,
            yy,
            active_shape.astype(float),
            levels=[0.5],
            colors=["#374151"],
            linewidths=0.45,
        )
        axis.set_title(rf"({panel}) {labels[method]}", loc="left")
        axis.text(
            0.04,
            0.96,
            rf"max error $={tex_sci(maxima[method])}$",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.4,
            color=BLACK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
        )
        axis.set_aspect("equal")
        axis.set_xlim(0.0, DOMAIN_SIZE)
        axis.set_ylim(0.0, DOMAIN_SIZE)
        axis.set_xlabel(r"$x$ (cm)")
        axis.grid(False)
        if method == "drm":
            axis.set_ylabel(r"$y$ (cm)")
        else:
            axis.set_yticklabels([])
    assert plot is not None
    colorbar = figure.colorbar(
        plot,
        ax=axes,
        location="bottom",
        fraction=0.065,
        pad=0.105,
        aspect=52,
    )
    colorbar.set_label("Normalized pointwise error (common logarithmic scale)", fontsize=10.8)
    colorbar.ax.tick_params(labelsize=9.5)
    save_figure(figure, output, "experiment4_flux_error_comparison")




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

    cube_progress = pd.read_csv(DATA / "ji2024_drm_validated_progress.csv")
    cube_terminal = pd.read_csv(DATA / "ji2024_drm_validated_terminal.csv")
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

    reference = np.load(EXPERIMENT4_RESULTS / "reference_subdiv48.npz")
    reference_keff = float(reference["keff"])
    reactor_methods = (
        ("drm_seed2026071101", "DRM", ORANGE),
        ("gipmnn_seed2026071102", "GIPMNN", GREEN),
        ("pc_gipmnn_seed2026071103", "PC-GIPMNN", PURPLE),
    )
    for directory, label, color in reactor_methods:
        rows = pd.read_csv(EXPERIMENT4_RESULTS / directory / "convergence.csv")
        epochs = rows["epoch"].to_numpy()
        keff_error = np.abs(rows["keff"].to_numpy() - reference_keff) / abs(reference_keff)
        flux_error = rows["flux_error"].to_numpy()
        summary = json.loads((EXPERIMENT4_RESULTS / directory / "summary.json").read_text(encoding="utf-8"))
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
    plot_experiment1(output)
    plot_experiment1_ablation(output)
    plot_experiment4_overview(output)
    plot_experiment4_errors(output)
    plot_appendix_convergence(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redraw the manuscript figures without touching scientific data.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "figure_refresh_20260808" / "candidate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_all(args.output.resolve())


if __name__ == "__main__":
    main()
