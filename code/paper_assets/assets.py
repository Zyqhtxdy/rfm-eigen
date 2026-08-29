from __future__ import annotations

import hashlib
import json
import shutil
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter

from paper_assets.paths import DATA, FIGURES, ROOT

DATA_DIR = DATA
FIG_DIR = FIGURES
NONLINEAR_RESULTS = (
    ROOT / "nonlinear_eigen_rfm_study" / "results" / "e7_e5_fair_comparison"
)


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
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def tex_sci(value: float) -> str:
    mantissa, exponent = f"{float(value):.2e}".split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"
def tex_measurement(median: float, minimum: float, maximum: float) -> str:
    # The range sits under the median rather than beside it.  Side by side, the
    # two error columns of Table 6 are wide enough to force the whole table
    # down to \scriptsize; stacked, the same numbers fit at \small.
    #
    # A nested tabular rather than \shortstack: the latter sets its baseline at
    # the foot of the stack, so the median climbed into the row above it.  The
    # [t] alignment keeps the median on its own row and hangs the range below.
    return (
        r"\begin{tabular}[t]{@{}r@{}}"
        rf"\({tex_sci(median)}\)\\"
        rf"\([{tex_sci(minimum)},{tex_sci(maximum)}]\)"
        r"\end{tabular}"
    )


def plot_experiment1() -> None:
    frame = pd.read_csv(
        DATA_DIR / "experiment1_theory_matched_summary.csv"
    ).sort_values("N")
    metadata = json.loads(
        (DATA_DIR / "experiment1_theory_matched_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    guides = json.loads(
        (DATA_DIR / "experiment1_rate_guides.json").read_text(encoding="utf-8")
    )
    draw_count = int(frame["samples"].iloc[0])
    budgets = frame["N"].to_numpy()
    gaps = frame["h1_gap_q90"].to_numpy()
    eigenvalue_errors = frame["cluster_relative_error_q90"].to_numpy()
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
    gap_slope = float(metadata["gap_q90_slope"])
    eigenvalue_slope = float(metadata["eigenvalue_q90_slope"])
    gap_interval = metadata["gap_q90_slope_bootstrap_95"]
    eigenvalue_interval = metadata[
        "eigenvalue_q90_slope_bootstrap_95"
    ]

    figure, axes = plt.subplots(1, 2, figsize=page_size(0.96, 2.55))
    axes[0].loglog(
        budgets,
        gaps,
        marker="o",
        color="#274c77",
        linewidth=1.4,
        markersize=4.0,
        label=r"$Q_{0.9}(\operatorname{gap}_{H^1})$",
    )
    axes[0].loglog(
        budgets,
        gap_reference,
        color="#222222",
        linestyle="--",
        linewidth=1.1,
        label=r"reference $C_HN^{-1/2}$",
    )
    axes[0].set_title(r"(a) $H^1$ eigenspace gap")
    axes[0].set_xlabel(r"number of features $N$")
    axes[0].set_ylabel(r"$Q_{0.9}(\operatorname{gap}_{H^1})$")
    axes[0].text(
        0.05,
        0.08,
        rf"observed $N^{{{gap_slope:.3f}}}$"
        + "\n"
        + rf"95% CI $[{gap_interval[0]:.3f},{gap_interval[1]:.3f}]$",
        transform=axes[0].transAxes,
        fontsize=8,
        va="bottom",
    )
    axes[0].legend(loc="upper right", frameon=False)

    axes[1].loglog(
        budgets,
        eigenvalue_errors,
        marker="^",
        color="#4f7f52",
        linewidth=1.4,
        markersize=4.2,
        label=r"$Q_{0.9}(e_{\rm cl})$",
    )
    axes[1].loglog(
        budgets,
        eigenvalue_reference,
        color="#222222",
        linestyle="--",
        linewidth=1.1,
        label=r"reference $C_\lambda N^{-1}$",
    )
    axes[1].set_title(r"(b) Eigenvalue error")
    axes[1].set_xlabel(r"number of features $N$")
    axes[1].set_ylabel(r"$Q_{0.9}(e_{\rm cl})$")
    axes[1].text(
        0.05,
        0.08,
        rf"observed $N^{{{eigenvalue_slope:.3f}}}$"
        + "\n"
        + rf"95% CI $[{eigenvalue_interval[0]:.3f},{eigenvalue_interval[1]:.3f}]$",
        transform=axes[1].transAxes,
        fontsize=8,
        va="bottom",
    )
    axes[1].legend(loc="upper right", frameon=False)

    tick_values = budgets.astype(float)
    # Eight labels do not fit across a panel of this width, so the ticks all
    # stay and every other budget is named.
    named = {0, 2, 5, len(tick_values) - 1}
    tick_labels = [
        str(int(value)) if index in named else ""
        for index, value in enumerate(tick_values)
    ]
    for axis in axes:
        axis.set_xticks(tick_values)
        axis.set_xticklabels(tick_labels)
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xlim(budgets[0] * 0.96, budgets[-1] * 1.04)
        axis.grid(True, which="major", color="#d7d7d7", linewidth=0.6)
    figure.tight_layout()
    figure.savefig(FIG_DIR / "experiment1_rewrite_overview.pdf")
    plt.close(figure)

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{Empirical error statistics for the exact double eigenvalue \(\lambda_7=\lambda_8\), based on {draw_count} independent feature draws at each \(N\).}}",
        r"\label{tab:experiment1_h1_middle}",
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"$N$ & $Q_{0.9}(\zeta_N(F_\star))$ & $Q_{0.9}(\operatorname{gap}_{H^1})$ & $Q_{0.9}(e_{\rm cl})$\\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            rf"{int(row['N'])} & "
            rf"\({tex_sci(row['eta_f_h1_q90'])}\) & "
            rf"\({tex_sci(row['h1_gap_q90'])}\) & "
            rf"\({tex_sci(row['cluster_relative_error_q90'])}\)\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text(
        DATA_DIR / "experiment1_rewrite_summary_table.tex", "\n".join(lines)
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


def plot_experiment2() -> None:
    """Cost against accuracy for the three methods on the unit ball.

    Errors rather than eigenvalues, because the spectrum is known in closed
    form and there is no fitted limit to draw; and both axes logarithmic,
    because the three curves are separated by decades rather than by factors.
    """
    run = _experiment2_run()
    figure, axis = plt.subplots(1, 1, figsize=page_size(0.78, 2.55))

    series = (
        ("P2 FEM", r"$P_2$ FEM", "#274c77", "o"),
        ("Eig-PIELM", "Eig-PIELM", "#4a7c59", "^"),
    )
    for key, label, color, marker in series:
        rows = _experiment2_rows(run, key)
        axis.loglog(
            [row["median_seconds"] for row in rows],
            [row["median_error"] for row in rows],
            marker=marker, color=color, linewidth=1.3, markersize=4.0,
            label=label,
        )

    rows = _experiment2_rows(run, "RFM")
    seconds = np.array([row["median_seconds"] for row in rows])
    median = np.array([row["median_error"] for row in rows])
    axis.errorbar(
        seconds, median,
        yerr=np.vstack([
            median - np.array([row["min_error"] for row in rows]),
            np.array([row["max_error"] for row in rows]) - median,
        ]),
        marker="s", color="#9a5b35", linewidth=1.3, markersize=4.0,
        capsize=3.0, label="RFM",
    )
    # Only the two ends of the sampled curve are named.  Labelling all five
    # budgets put text on the curve it labelled and crowded the last two
    # together; the budgets themselves are the second column of Table 2, and
    # what the figure has to say is that the curve runs from one corner to the
    # other while the baselines do not.
    # both to the right of their markers, which is empty on either end
    for index, offset, align in ((0, (8, 0), "left"), (-1, (8, 0), "left")):
        axis.annotate(
            rf"$N={int(rows[index]['features'])}$",
            (seconds[index], median[index]),
            textcoords="offset points", xytext=offset, ha=align,
            va="center", fontsize=7.5, color="#9a5b35",
        )

    axis.set_xlabel("time (s)")
    axis.set_ylabel("maximum relative error")
    # Limits from the data rather than fixed: the range the three curves cover
    # depends on the problem, and a window sized for a different one leaves
    # empty decades that flatten everything into the top of the frame.
    times = [row["median_seconds"] for row in run["rows"]]
    errors = [row["median_error"] for row in run["rows"]]
    errors += [row["min_error"] for row in _experiment2_rows(run, "RFM")]
    axis.set_xlim(min(times) / 2.0, max(times) * 3.0)
    axis.set_ylim(min(errors) / 5.0, max(errors) * 3.0)
    axis.legend(frameon=False, loc="lower left")
    axis.grid(True, which="major", color="#d7d7d7", linewidth=0.6)
    figure.tight_layout()
    figure.savefig(FIG_DIR / "experiment2_error_versus_time.pdf")
    figure.savefig(FIG_DIR / "experiment2_error_versus_time.png", dpi=240)
    plt.close(figure)


def build_experiment2_table() -> None:
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
            r"\caption{The first ten eigenvalues of the radially graded unit ball, for "
            r"the isoparametric \(P_2\) finite element method, Eig-PIELM, and "
            rf"the RFM. RFM entries are medians over {_spelled_count(draws)} "
            r"independent feature draws, with the observed range in brackets.}"
        ),
        r"\label{tab:ball_benchmark_summary}",
        r"\begin{tabular}{llrl}",
        r"\toprule",
        r"method & resolution & time (s) & maximum relative error\\",
        r"\midrule",
    ]
    for row in _experiment2_rows(run, "P2 FEM"):
        lines.append(
            rf"\(P_2\) FEM & \({row['setting']}\) & "
            rf"{row['median_seconds']:.2f} & "
            rf"\({tex_sci(row['median_error'])}\)\\"
        )
    lines.append(r"\midrule")
    for row in _experiment2_rows(run, "Eig-PIELM"):
        lines.append(
            rf"Eig-PIELM & \(\deg={row['degree']}\) & "
            rf"{row['median_seconds']:.2f} & "
            rf"\({tex_sci(row['median_error'])}\)\\"
        )
    lines.append(r"\midrule")
    for row in _experiment2_rows(run, "RFM"):
        lines.append(
            rf"RFM & \(N={int(row['features'])}\) & "
            rf"{row['median_seconds']:.2f} & "
            rf"\({tex_sci(row['median_error'])}\,["
            rf"{tex_sci(row['min_error'])},{tex_sci(row['max_error'])}]\)\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text(
        DATA_DIR / "experiment2_rewrite_summary_table.tex", "\n".join(lines)
    )


#: Draw counts the captions spell out rather than print as numerals.
_NUMBER_WORDS = {3: "three", 5: "five", 10: "ten", 20: "twenty"}


def _spelled_count(count: int) -> str:
    return _NUMBER_WORDS.get(count, str(count))


#: The random feature run the Example 3 entries are taken from.  The archived
#: aggregation holds the five draws of the original run and is left as it
#: stands; this run repeats the same configuration over twenty draws.
EXPERIMENT3_RFM_RUN = DATA_DIR / "runs" / "experiment3"


def _experiment3_rfm() -> pd.DataFrame:
    """The random feature side of Example 3, over as many draws as were run.

    Only the first three ordered values are kept: they are what the source
    reports, and what the table and the figure carry.
    """
    summary = EXPERIMENT3_RFM_RUN / "rfm_summary.csv"
    if not summary.exists():
        return pd.read_csv(DATA_DIR / "experiment3_qmc_common_budget_eigs.csv")
    frame = pd.read_csv(summary)
    return frame[frame["eigen_index"] <= 3].reset_index(drop=True)


def build_experiment3_table() -> None:
    """Table 3, with both methods measured by one quadrature rule.

    The Deep Ritz eigenvalue is a Rayleigh quotient of a trained network, and
    the value reported here is that quotient re-evaluated on the rule the RFM
    assembles with.  The networks are the stored ones; nothing is retrained,
    and the re-evaluation lowers every Deep Ritz error, so the comparison is
    not made favourable by the change.
    """
    frame = pd.read_csv(DATA_DIR / "experiment3_first3_comparison.csv")
    neural = pd.read_csv(DATA_DIR / "experiment3_drm_quadrature_reevaluation.csv")
    key = {
        (row["potential"], int(row["mode"])): float(row["rel_error_beta22"])
        for _, row in neural.iterrows()
    }
    rfm = _experiment3_rfm()
    reported = rfm[rfm["N"] == int(frame["rfm_size"].iloc[0])]
    rfm_error = {
        (row["potential"], int(row["eigen_index"])): float(row["rel_error_median"])
        for _, row in reported.iterrows()
    }
    draws = _spelled_count(int(reported["trials"].iloc[0]))
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{Relative errors of the first three ordered eigenvalues for the ten-dimensional benchmark. RFM entries are medians over {draws} independent feature draws.}}",
        r"\label{tab:tenD_summary_new}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"potential & eigenvalue & DRM error & RFM error\\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        potential = r"$t^2$" if row["potential"] == "square" else r"$e^{-\pi t}$"
        index = int(row["eigen_index"])
        lines.append(
            rf"{potential} & $\lambda_{{{index}}}$ & "
            rf"\({tex_sci(key[(row['potential'], index)])}\) & "
            rf"\({tex_sci(rfm_error[(row['potential'], index)])}\)\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text(
        DATA_DIR / "experiment3_rewrite_summary_table.tex", "\n".join(lines)
    )


def plot_experiment3() -> None:
    """Approximated eigenvalues against the exact levels, as in Ji et al. Fig. 7.

    Their figure follows the three eigenvalue estimates through training; the
    RFM has no training, so the estimates are followed through the feature
    count instead.  Plotting the values rather than their errors is what makes
    the multiple level visible: the second and third ordered Ritz values
    approach one and the same line.
    """
    eigs = _experiment3_rfm()
    neural = pd.read_csv(DATA_DIR / "experiment3_drm_quadrature_reevaluation.csv")
    figure, axes = plt.subplots(1, 2, figsize=page_size(0.96, 2.45))
    titles = {"square": r"(a) $V(t)=t^{2}$", "exp": r"(b) $V(t)=e^{-\pi t}$"}
    markers = {1: "o", 2: "s", 3: "^"}
    colors = {1: "#274c77", 2: "#9a5b35", 3: "#4f7f52"}

    for axis, potential in zip(axes, ("square", "exp")):
        subset = eigs[eigs["potential"] == potential]
        budgets = np.sort(subset["N"].unique())
        for level in sorted(subset["lambda_ref"].unique()):
            axis.axhline(level, color="#555555", linestyle="--", linewidth=1.0)
        for index in (1, 2, 3):
            part = subset[subset["eigen_index"] == index].sort_values("N")
            median = part["lambda_median"].to_numpy()
            axis.plot(
                part["N"].to_numpy(),
                median,
                marker=markers[index],
                color=colors[index],
                linewidth=1.3,
                markersize=4.0,
                label=rf"$\lambda_{index}$",
            )
        # The Deep Ritz estimates carry the colour and the shape of the level
        # they approximate, so that each one can be read against its own line;
        # they are drawn open, and off the feature axis, because the method
        # has no feature count.
        row = neural[neural["potential"] == potential].sort_values("mode")
        offset = budgets[-1] + 0.20 * (budgets[-1] - budgets[0])
        for _, entry in row.iterrows():
            index = int(entry["mode"])
            axis.plot(
                [offset],
                [float(entry["lambda_beta22"])],
                linestyle="none",
                marker=markers[index],
                markerfacecolor="none",
                markeredgecolor=colors[index],
                markersize=5.0,
                markeredgewidth=1.6,
                label="DRM" if index == 1 else None,
            )
        axis.set_title(titles[potential], loc="left")
        axis.set_xlabel(r"number of features $N$")
        axis.set_xticks(list(budgets) + [offset])
        axis.set_xticklabels([str(int(value)) for value in budgets] + ["DRM"])
        axis.set_xlim(budgets[0] - 40, offset + 45)
        axis.grid(True, which="major", color="#e2e2e2", linewidth=0.6)
    axes[0].set_ylabel(r"eigenvalue")
    axes[0].legend(frameon=False, loc="center right", ncol=1, fontsize=8)
    figure.tight_layout()
    figure.savefig(FIG_DIR / "experiment3_qmc_errors.pdf")
    figure.savefig(FIG_DIR / "experiment3_qmc_errors.png", dpi=240)
    plt.close(figure)
def build_experiment4_assets() -> None:
    from iaea_repro.paper_assets import generate_experiment4_assets

    # the builder resolves its own directories under the root it is given;
    # the figures of this repository live beside code, not inside it
    generate_experiment4_assets(ROOT, figure_directory=FIGURES)


def verify_nonlinear_artifacts() -> None:
    manifest = NONLINEAR_RESULTS / "SHA256SUMS"
    recorded: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", maxsplit=1)
        recorded[name] = digest
    for name, digest in recorded.items():
        path = NONLINEAR_RESULTS / name
        content = path.read_bytes()
        if path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
            content = content.replace(b"\r\n", b"\n")
        if hashlib.sha256(content).hexdigest() != digest:
            raise RuntimeError(f"Nonlinear artifact hash mismatch: {name}")


def build_experiment5_table() -> None:
    """Table 6, from the timed comparison run.

    Both methods are timed in one session at one thread, and the timer covers
    mesh or basis construction, assembly and solve; error evaluation is
    outside it.  The weak Galerkin rows are labelled by mesh, as in the source
    work and in the text, so that N keeps its single meaning as the number of
    random features.
    """
    baseline = pd.read_csv(DATA_DIR / "experiment5_wg_timed.csv")
    trials = pd.read_csv(DATA_DIR / "experiment5_rfm_timed.csv")
    baseline = baseline.loc[baseline["converged"].astype(bool)]
    trials = trials.loc[trials["converged"].astype(bool)]
    draws = _spelled_count(int(trials["seed"].nunique()))
    repeats = int(trials.groupby(["features", "seed"]).size().median())

    def entry(group: pd.DataFrame, resolution: str, method: str) -> str:
        # The weak Galerkin scheme is deterministic, so only its timing varies
        # over the repeats; a bracket on its errors would repeat one number.
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
                tex_measurement(
                    per_draw[column].median(),
                    per_draw[column].min(),
                    per_draw[column].max(),
                )
                for column in ("lambda_abs_error", "energy_abs_error")
            )
        return (
            rf"{method} & {resolution} & "
            rf"{seconds.median():.2f}\,[{seconds.min():.2f},{seconds.max():.2f}] & "
            rf"{errors}\\"
        )

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{The shifted Gross--Pitaevskii problem, for the weak Galerkin scheme and the RFM. RFM entries are medians over {draws} independent feature draws, each solved {_spelled_count(repeats)} times for the clock.}}",
        r"\label{tab:nonlinear_gpe_comparison}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"method & resolution & time (s) & "
        r"\(\lvert\lambda-\lambda_{\rm ref}\rvert\) & \(\lvert E-E_{\rm ref}\rvert\)\\",
        r"\midrule",
    ]
    for size in sorted(baseline["subdivisions"].unique()):
        group = baseline.loc[baseline["subdivisions"] == size]
        lines.append(entry(group, rf"\(h=1/{int(size)}\)", "WG"))
    lines.append(r"\midrule")
    for features in sorted(trials["features"].unique()):
        group = trials.loc[trials["features"] == features]
        lines.append(entry(group, rf"\(N={int(features)}\)", "RFM"))
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text(DATA_DIR / "experiment5_nonlinear_gpe_table.tex", "\n".join(lines))


def build_experiment6_table() -> None:
    """Table 9, from the timed comparison run.

    A GFLM--KTM entry is the median over the published initial pairs that
    converge at that mesh, since Li et al. give no rule for choosing one and
    the cost varies by a factor of five across pairs that reach the same
    state.  An RFM entry is the median over the feature draws and repeats.
    Calls that did not converge produced no solution and are excluded from
    both medians.
    """
    baseline = pd.read_csv(DATA_DIR / "experiment6_gflm_ktm_timed.csv")
    trials = pd.read_csv(DATA_DIR / "experiment6_rfm_timed.csv")
    baseline = baseline.loc[baseline["converged"].astype(bool)]
    trials = trials.loc[trials["converged"].astype(bool)]

    def entry(group: pd.DataFrame, resolution: str, method: str) -> str:
        seconds = group["wall_seconds"]
        return (
            rf"{method} & {resolution} & "
            rf"{seconds.median():.1f}\,[{seconds.min():.1f},{seconds.max():.1f}] & "
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
        r"\caption{Errors and computation times of GFLM--KTM and the RFM for the rotating two-component dipolar condensate. RFM entries are medians over ten independent feature draws and GFLM--KTM entries over twenty published initial states, with the observed range in brackets.}",
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
    write_text(DATA_DIR / "experiment6_dipolar_bec_table.tex", "\n".join(lines))


def build_nonlinear_figures() -> None:
    verify_nonlinear_artifacts()
    shutil.copyfile(
        NONLINEAR_RESULTS / "e7_e5_error_comparison.pdf",
        FIG_DIR / "experiment5_6_nonlinear_error_comparison.pdf",
    )
    shutil.copyfile(
        NONLINEAR_RESULTS / "e5_density_comparison.pdf",
        FIG_DIR / "experiment6_dipolar_density_comparison.pdf",
    )


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    use_plot_style()
    plot_experiment1()
    build_experiment2_table()
    plot_experiment2()
    build_experiment3_table()
    plot_experiment3()
    build_experiment4_assets()
    # the five-panel appendix figure this used to build was split in two and is
    # now drawn by redesign_figures; calling it here only writes a dead file
    build_experiment5_table()
    build_experiment6_table()
    build_nonlinear_figures()


if __name__ == "__main__":
    main()
