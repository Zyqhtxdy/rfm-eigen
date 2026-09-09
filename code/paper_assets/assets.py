from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullLocator

from paper_assets.formatting import tex_time_seconds
from paper_assets.paths import DATA, FIGURES, REPO_DATA, ROOT
from paper_assets.records import converged_rows, repetitions_per_draw

DATA_DIR = REPO_DATA
TABLE_DIR = DATA
FIG_DIR = FIGURES


def use_plot_style() -> None:
    plt.rcParams.update(
        {
            # STIXGeneral is Times-metric-compatible, so figure text
            # matches the body face rather than sitting beside it
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            # Each figure below is drawn at the width it is finally shown at,
            # so these are the point sizes the reader sees.  Drawing wide and
            # letting \includegraphics shrink the result was what put every
            # label under 6 pt against a 10 pt caption.
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            # Type 3 is matplotlib's default and is refused by most
            # typesetters; 42 embeds TrueType outlines instead
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 140,
            # anything rasterized inside a figure is written at print
            # resolution rather than at the screen resolution above
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
        }
    )


#: Width of the manuscript's text block, in inches.  ``\the\textwidth`` is
#: 384 pt, and a TeX point is 1/72.27 in.
TEXT_WIDTH = 384.0 / 72.27


def page_size(fraction: float, height: float) -> tuple[float, float]:
    """Canvas for a figure included at ``fraction`` of the text width.

    Returning the true printed width means ``\\includegraphics`` neither
    enlarges nor reduces the result, so nothing set in points is rescaled on
    its way to the page.
    """
    return (fraction * TEXT_WIDTH, height)


def write_text(path: Path, content: str) -> None:
    """Write a generated file with the newlines .gitattributes expects, so
    that a rebuild on Windows does not differ from the committed copy by
    line endings alone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content.rstrip() + "\n")


def tex_sci(value: float) -> str:
    mantissa, exponent = f"{float(value):.2e}".split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def plot_experiment1() -> None:
    frame = pd.read_csv(DATA_DIR / "experiment1_theory_matched_summary.csv").sort_values("N")
    guides = json.loads(
        (DATA_DIR / "experiment1_rate_guides.json").read_text(encoding="utf-8")
    )
    budgets = frame["N"].to_numpy()
    # The arithmetic mean over the draws at each budget. The quadrature is
    # resolved to the point where the three orders that were compared agree in
    # the fitted slope to six decimals, so every draw carries the rate and the
    # mean is not held up by an unresolved one.
    gaps = frame["h1_gap_mean"].to_numpy()
    eigenvalue_errors = frame["cluster_relative_error_mean"].to_numpy()
    gap_reference = float(guides["gap_constant"]) * budgets ** (-0.5)
    eigenvalue_reference = float(guides["eigenvalue_constant"]) * budgets ** (-1.0)
    gap_ratio = gaps / gap_reference
    eigenvalue_ratio = eigenvalue_errors / eigenvalue_reference
    # The manuscript makes no claim that the data stay under the calibrated
    # references, so exceeding them is not an error here; it is only worth
    # knowing about, because it would mean the calibration no longer describes
    # the runs it was fitted on.
    if np.any(gap_ratio > 1.0) or np.any(eigenvalue_ratio > 1.0):
        warnings.warn(
            "Example 1 data exceed the calibrated rate reference",
            RuntimeWarning,
            stacklevel=2,
        )
    # Margins are fixed rather than left to tight_layout, because the shape of
    # the axes box is what a reader reads a slope from: a flatter box makes the
    # same exponent look gentler. Height over width is held at 0.714.
    left, right, top, bottom, wspace = 0.105, 0.985, 0.895, 0.225, 0.30
    width = page_size(0.96, 1.0)[0]
    panel_width = width * (right - left) / (2 + wspace)
    figure, axes = plt.subplots(
        1, 2, figsize=(width, 0.714 * panel_width / (top - bottom))
    )
    figure.subplots_adjust(
        left=left, right=right, top=top, bottom=bottom, wspace=wspace
    )
    axes[0].loglog(
        budgets,
        gaps,
        marker="o",
        color="#274c77",
        linewidth=1.4,
        markersize=4.0,
    )
    axes[0].loglog(
        budgets,
        gap_reference,
        color="#222222",
        linestyle="--",
        linewidth=1.1,
        label=r"reference $O(N^{-1/2})$",
    )
    # The panel title and the caption already name the quantity, and the caption
    # carries the statistic, so an axis label would state the same thing a third
    # time in the one notation a reader has to decode.
    axes[0].set_title(r"(a) $H^1$ eigenspace gap")
    axes[0].set_xlabel(r"number of features $N$")
    axes[0].legend(loc="lower left", frameon=False)

    axes[1].loglog(
        budgets,
        eigenvalue_errors,
        marker="o",
        color="#4f7f52",
        linewidth=1.4,
        markersize=4.0,
    )
    axes[1].loglog(
        budgets,
        eigenvalue_reference,
        color="#222222",
        linestyle="--",
        linewidth=1.1,
        label=r"reference $O(N^{-1})$",
    )
    axes[1].set_title(r"(b) Eigenvalue error")
    axes[1].set_xlabel(r"number of features $N$")
    axes[1].legend(loc="lower left", frameon=False)

    # Powers of two are evenly spaced on this scale and few enough to fit;
    # the markers show where the eight budgets themselves fall, and Table 1
    # lists them.
    ticks = [128.0, 256.0, 512.0, 1024.0]
    for axis in axes:
        axis.set_xticks(ticks)
        axis.set_xticklabels([str(int(value)) for value in ticks])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_xlim(budgets[0] * 0.96, ticks[-1] * 1.03)
        axis.grid(True, which="major", color="#d7d7d7", linewidth=0.6)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIG_DIR / "experiment1_rewrite_overview.pdf")
    plt.close(figure)

    build_experiment1_table()


def build_experiment1_table(*, output_directory: Path | None = None) -> None:
    """Build Table 1 independently of its figure, for read-only auditing."""
    frame = pd.read_csv(DATA_DIR / "experiment1_theory_matched_summary.csv").sort_values("N")
    draw_count = int(frame["samples"].iloc[0])
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{Errors for the exact double eigenvalue \(\lambda_7=\lambda_8\).  Each entry is the arithmetic mean over {draw_count} independent feature draws.}}",
        r"\label{tab:experiment1_h1_middle}",
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"$N$ & $\zeta_N(F_\star)$ & $d_{H^1}(E_\star,E_{\star,N})$ & $e_{\rm cl}$\\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            rf"{int(row['N'])} & "
            rf"\({tex_sci(row['eta_f_h1_mean'])}\) & "
            rf"\({tex_sci(row['h1_gap_mean'])}\) & "
            rf"\({tex_sci(row['cluster_relative_error_mean'])}\)\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text(
        (output_directory or TABLE_DIR) / "experiment1_rewrite_summary_table.tex", "\n".join(lines)
    )


def _experiment2_run() -> dict:
    """The recorded graded unit-ball measurement.

    Taken with nothing else on the machine, so the times are comparable across
    the three methods; regenerating it under load would move every second in
    the table without moving any error.
    """
    path = DATA_DIR / "experiment2_graded_ball_run.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _experiment2_rows(run: dict, method: str) -> list[dict]:
    return [row for row in run["rows"] if row["method"] == method]



def build_experiment2_table(*, output_directory: Path | None = None) -> None:
    """The three cost-accuracy curves as a table.

    The dimension column is the size of the eigenproblem each method actually
    solves -- finite element unknowns, the admissible subspace of the
    collocation baseline, and the retained rank of the sampled space -- so the
    accuracies are read against comparable amounts of work.
    """
    run = _experiment2_run()
    draws = _experiment2_rows(run, "RFM")[0]["draws"]
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        (
            r"\caption{Maximum relative errors over the first ten eigenvalues "
            r"and computation times of the isoparametric \(P_2\) finite element "
            r"method, Eig-PIELM, and the RFM for the radially graded unit ball. "
            r"For the finite element method, \(n\) denotes the number of mesh refinements. "
            rf"RFM entries are medians over {_spelled_count(draws)} independent "
            r"feature draws.}"
        ),
        r"\label{tab:ball_benchmark_summary}",
        r"\begin{tabular}{llrl}",
        r"\toprule",
        r"method & resolution & time (s) & maximum relative error\\",
        r"\midrule",
    ]
    for row in _experiment2_rows(run, "P2 FEM"):
        lines.append(
            rf"\(P_2\) FEM & \(n={int(row['subdivisions'])}\) & "
            rf"{tex_time_seconds(row['median_seconds'])} & "
            rf"\({tex_sci(row['median_error'])}\)\\"
        )
    lines.append(r"\midrule")
    for row in _experiment2_rows(run, "Eig-PIELM"):
        lines.append(
            rf"Eig-PIELM & \(\deg={row['degree']}\) & "
            rf"{tex_time_seconds(row['median_seconds'])} & "
            rf"\({tex_sci(row['median_error'])}\)\\"
        )
    lines.append(r"\midrule")
    for row in _experiment2_rows(run, "RFM"):
        lines.append(
            rf"RFM & \(N={int(row['features'])}\) & "
            rf"{tex_time_seconds(row['median_seconds'])} & "
            rf"\({tex_sci(row['median_error'])}\)\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text(
        (output_directory or TABLE_DIR) / "experiment2_rewrite_summary_table.tex", "\n".join(lines)
    )


#: Draw counts the captions spell out rather than print as numerals.
_NUMBER_WORDS = {3: "three", 5: "five", 10: "ten", 20: "twenty"}


def _spelled_count(count: int) -> str:
    return _NUMBER_WORDS.get(count, str(count))


#: The random feature run the Example 3 entries are taken from.  The archived
#: aggregation holds the five draws of the original run and is left as it
#: stands; this run repeats the same configuration over twenty draws.
EXPERIMENT3_RFM_RUN = DATA_DIR / "runs" / "experiment3_independent"


def _experiment3_rfm() -> pd.DataFrame:
    """The random feature side of Example 3, over as many draws as were run.

    Only the first three ordered values are kept: they are what the source
    reports, and what the table and the figure carry.
    """
    frame = pd.read_csv(EXPERIMENT3_RFM_RUN / "rfm_summary.csv")
    return frame[frame["eigen_index"] <= 3].reset_index(drop=True)


def plot_experiment3() -> None:
    """Approximated eigenvalues against the exact levels, as in Ji et al. Fig. 7.

    Their figure follows the three eigenvalue estimates through training; the
    RFM has no training, so the estimates are followed through the feature
    count instead.  Plotting the values rather than their errors is what makes
    the multiple level visible: the second and third ordered Ritz values
    approach one and the same line.
    """
    eigs = _experiment3_rfm()
    neural = pd.read_csv(DATA_DIR / "experiment3_drm_independent.csv")
    figure = plt.figure(figsize=page_size(0.96, 2.46))
    outer = figure.add_gridspec(1, 2, wspace=0.34, left=0.10, right=0.98,
                                top=0.905, bottom=0.285)
    titles = {"square": r"(a) $V(t)=t^{2}$", "exp": r"(b) $V(t)=e^{-\pi t}$"}
    markers = {1: "o", 2: "s", 3: "^"}
    colors = {1: "#274c77", 2: "#9a5b35", 3: "#4f7f52"}
    handles: list = []
    labels: list = []

    for column, potential in enumerate(("square", "exp")):
        # The Deep Ritz estimates have no feature count, so they are given a
        # strip of their own rather than a tick invented on the feature axis.
        inner = outer[column].subgridspec(1, 2, width_ratios=[1.0, 0.14],
                                          wspace=0.07)
        axis = figure.add_subplot(inner[0])
        strip = figure.add_subplot(inner[1], sharey=axis)

        subset = eigs[eigs["potential"] == potential]
        budgets = np.sort(subset["N"].unique())
        # the reference levels run across the strip as well, so that each open
        # marker can still be read against the line it approximates
        for level in sorted(subset["lambda_ref"].unique()):
            for target in (axis, strip):
                target.axhline(level, color="#555555", linestyle="--",
                               linewidth=1.0)
        for index in (1, 2, 3):
            part = subset[subset["eigen_index"] == index].sort_values("N")
            axis.plot(
                part["N"].to_numpy(),
                part["lambda_median"].to_numpy(),
                marker=markers[index],
                color=colors[index],
                linewidth=1.3,
                markersize=4.0,
                label=rf"$\lambda_{index}$",
            )
        # The estimates carry the colour and the shape of the level they
        # approximate, and are drawn open.  Within the strip they are spread by
        # mode, because two of them differ by less than a tenth of a percent of
        # the axis and would otherwise coincide; the ordinate is untouched and
        # the strip carries no scale, so the spread states nothing.
        row = neural[neural["potential"] == potential].sort_values("mode")
        for _, entry in row.iterrows():
            index = int(entry["mode"])
            strip.plot(
                [(index - 2) * 0.52],
                [float(entry["lambda_final"])],
                linestyle="none",
                marker=markers[index],
                markerfacecolor="none",
                markeredgecolor=colors[index],
                markersize=4.4,
                markeredgewidth=1.35,
            )

        axis.set_title(titles[potential], loc="left")
        axis.set_xlabel(r"number of features $N$")
        axis.set_xticks(list(budgets))
        axis.set_xticklabels([str(int(value)) for value in budgets])
        axis.set_xlim(budgets[0] - 40, budgets[-1] + 40)
        axis.grid(True, which="major", color="#e2e2e2", linewidth=0.6)

        strip.set_xticks([0.0])
        strip.set_xticklabels(["DRM"])
        strip.set_xlim(-1.05, 1.05)
        strip.grid(False)
        strip.tick_params(labelleft=False, left=False)
        strip.spines["left"].set_linewidth(0.8)
        strip.spines["left"].set_color("#999999")

        if column == 0:
            axis.set_ylabel(r"eigenvalue")
            handles, labels = axis.get_legend_handles_labels()

    # one key for both panels: the right panel had no colour key of its own
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                  bbox_to_anchor=(0.54, 0.008), columnspacing=2.4,
                  handlelength=2.0)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIG_DIR / "experiment3_qmc_errors.pdf")
    figure.savefig(FIG_DIR / "experiment3_qmc_errors.png", dpi=240)
    plt.close(figure)
def build_experiment4_assets() -> None:
    from iaea_repro.paper_assets import generate_experiment4_assets

    # the builder resolves its own directories under the root it is given;
    # the figures of this repository live beside code, not inside it
    generate_experiment4_assets(ROOT, figure_directory=FIGURES, data_directory=TABLE_DIR)



def build_experiment5_table(*, output_directory: Path | None = None) -> None:
    """Table 5, from the timed comparison run.

    Both methods are timed in one session at one thread, and the timer covers
    mesh or basis construction, assembly and solve; error evaluation is
    outside it.  The weak Galerkin rows are labelled by mesh, as in the source
    work and in the text, so that N keeps its single meaning as the number of
    random features.
    """
    baseline = pd.read_csv(DATA_DIR / "experiment5_wg_timed.csv")
    trials = pd.read_csv(DATA_DIR / "experiment5_rfm_timed.csv")
    baseline = converged_rows(baseline)
    trials = converged_rows(trials)
    sample_counts = trials.groupby("features")["seed"].nunique().sort_index()
    feature_budgets = ",".join(str(int(size)) for size in sample_counts.index)
    sample_sizes = ", ".join(str(int(count)) for count in sample_counts)
    repeats = repetitions_per_draw(trials)

    def entry(group: pd.DataFrame, resolution: str, method: str) -> str:
        # The weak Galerkin scheme is deterministic, so its errors do not
        # vary over the repeats.
        seconds = group["wall_seconds"]
        if method == "WG":
            errors = " & ".join(
                rf"\({tex_sci(group[column].median())}\)"
                for column in ("lambda_abs_error", "energy_abs_error")
            )
        else:
            # A repeat re-times one draw; it does not resolve the problem
            # again.  The error statistics therefore run over the draws, and
            # only the timing runs over the calls.
            per_draw = group.drop_duplicates(subset="seed")
            errors = " & ".join(
                rf"\({tex_sci(per_draw[column].median())}\)"
                for column in ("lambda_abs_error", "energy_abs_error")
            )
        return (
            rf"{method} & {resolution} & "
            rf"{tex_time_seconds(seconds.median())} & "
            rf"{errors}\\"
        )

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{The Gross--Pitaevskii problem in Example~\ref{{ex:gpe}}, for the weak Galerkin scheme and the RFM. RFM entries are medians over the converged draws; the sample sizes for \(N={feature_budgets}\) are {sample_sizes}, respectively, with {_spelled_count(repeats)} timed repetitions per draw.}}",
        r"\label{tab:nonlinear_gpe_comparison}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"method & resolution & time (s) & "
        r"\(\lvert\lambda-\lambda_{\rm ref}\rvert\) & \(\lvert E-E_{\rm ref}\rvert\)\\",
        r"\midrule",
    ]
    for size in sorted(baseline["subdivisions"].unique()):
        group = baseline.loc[baseline["subdivisions"] == size]
        lines.append(entry(group, rf"\(n={int(size)}\)", "WG"))
    lines.append(r"\midrule")
    for features in sorted(trials["features"].unique()):
        group = trials.loc[trials["features"] == features]
        lines.append(entry(group, rf"\(N={int(features)}\)", "RFM"))
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text((output_directory or TABLE_DIR) / "experiment5_nonlinear_gpe_table.tex", "\n".join(lines))


def build_experiment6_table(*, output_directory: Path | None = None) -> None:
    """Table 6, from the timed comparison run.

    A GFLM--KTM entry is the median over the published initial pairs that
    converge at that mesh, since Li et al. give no rule for choosing one and
    the cost varies by a factor of five across pairs that reach the same
    state.  An RFM entry is the median over the feature draws and repeats.
    Calls that did not converge produced no solution and are excluded from
    both medians.
    """
    baseline = pd.read_csv(DATA_DIR / "experiment6_gflm_ktm_timed.csv")
    trials = pd.read_csv(DATA_DIR / "experiment6_rfm_timed.csv")
    baseline = converged_rows(baseline)
    trials = converged_rows(trials)

    def entry(group: pd.DataFrame, resolution: str, method: str) -> str:
        seconds = group["wall_seconds"]
        return (
            rf"{method} & {resolution} & "
            rf"{tex_time_seconds(seconds.median())} & "
            rf"\({tex_sci(group['energy_abs_error'].median())}\) & "
            rf"\({tex_sci(group['mu1_abs_error'].median())}\) & "
            rf"\({tex_sci(group['mu2_abs_error'].median())}\) & "
            rf"\({tex_sci(group['virial_residual'].median())}\)\\"
        )

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Errors and computation times of GFLM--KTM and the RFM for the rotating two-component dipolar condensate. RFM entries are medians over ten independent feature draws and GFLM--KTM entries over twenty initial states from the product of the ten published ones.}",
        r"\label{tab:dipolar_bec_comparison}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"method & resolution & time (s) & "
        r"\(\lvert E-E_{\rm ref}\rvert\) & \(\lvert\mu_1-\mu_{1,\rm ref}\rvert\) & "
        r"\(\lvert\mu_2-\mu_{2,\rm ref}\rvert\) & \(\lvert I_g\rvert\)\\",
        r"\midrule",
    ]
    for width in sorted(baseline["mesh_width"].unique(), reverse=True):
        group = baseline.loc[np.isclose(baseline["mesh_width"], width)]
        text = "1" if np.isclose(width, 1.0) else f"1/{int(round(1.0 / width))}"
        lines.append(entry(group, rf"\(h={text}\)", "GFLM--KTM"))
    lines.append(r"\midrule")
    for features in sorted(trials["features"].unique()):
        group = trials.loc[trials["features"] == features]
        lines.append(entry(group, rf"\(N={int(features)}\)", "RFM"))
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text((output_directory or TABLE_DIR) / "experiment6_dipolar_bec_table.tex", "\n".join(lines))



def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    use_plot_style()
    plot_experiment1()
    build_experiment2_table()
    plot_experiment3()
    build_experiment4_assets()
    # the five-panel appendix figure this used to build was split in two and is
    # now drawn by redesign_figures; calling it here only writes a dead file
    build_experiment5_table()
    build_experiment6_table()


if __name__ == "__main__":
    main()
