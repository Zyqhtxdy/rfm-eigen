from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patheffects import withStroke

from paper_assets.paths import EXPERIMENT4_RESULTS, ROOT

#: The random feature run the RFM row of the table is taken from.  The archived
#: audit holds the five draws of the original run and is left as it stands; this
#: run repeats that configuration over twenty, which is what the table reports.
EXPERIMENT4_RFM_RUN = ROOT / "data" / "runs" / "experiment4"

from .geometry import (
    DOMAIN_SIZE,
    build_collocation,
    cell_labels,
    labels_at,
)
from .metrics import mean_power_normalize, relative_flux_max_error
from .networks import ResidualFluxNet
from .plotting import plot_baseline_curves
from .reference import load_reference
from .rfm import evaluate_random_feature_field
from .trainer import _evaluate_rayleigh, _TensorCollocation

# Type 3 is matplotlib's default and is refused by most typesetters;
# 42 embeds TrueType outlines instead.  The figures below are drawn at
# the width they are printed at, so these point sizes are the ones the
# reader sees, and anything rasterized is written at print resolution.
plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "savefig.dpi": 600,
    }
)

#: Width of the manuscript's text block in inches; ``\the\textwidth``
#: is 384 pt and a TeX point is 1/72.27 in.
TEXT_WIDTH = 384.0 / 72.27


#: The draw counts the captions spell out; the sibling tables spell theirs too.
_NUMBER_WORDS = {5: "five", 10: "ten", 20: "twenty"}


def _spelled(count: int) -> str:
    return _NUMBER_WORDS.get(count, str(count))


def _rfm_representative(result_root: Path):
    """The draw the field figure shows, and the modes and coefficients it needs.

    The appendix states the rule -- the realization nearest the two componentwise
    median errors -- so the rule is applied here rather than left to a stored
    choice that a later change in the number of draws would silently invalidate.
    Distance is taken relatively, since the two errors differ by two orders.
    """
    rows_file = EXPERIMENT4_RFM_RUN / "rfm_rows.csv"
    summary_file = EXPERIMENT4_RFM_RUN / "rfm_summary.csv"
    if not (rows_file.exists() and summary_file.exists()):
        stored = np.load(result_root / "rfm_random_fourier" / "solution.npz")
        summary = json.loads(
            (result_root / "rfm_random_fourier" / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        return stored["modes"], stored["coefficients"], summary

    with rows_file.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with summary_file.open(encoding="utf-8") as handle:
        totals = next(r for r in csv.DictReader(handle) if r["method"] == "rfm")
    median_keff = float(totals["keff_error"])
    median_flux = float(totals["flux_error"])
    nearest = min(
        rows,
        key=lambda row: abs(float(row["keff_error"]) / median_keff - 1.0)
        + abs(float(row["flux_error"]) / median_flux - 1.0),
    )
    seed = int(nearest["seed"])
    stored = np.load(EXPERIMENT4_RFM_RUN / "states" / f"seed{seed}.npz")
    return (
        stored["modes"],
        stored["coefficients"],
        {
            "seed": seed,
            "representative_flux_error": float(nearest["flux_error"]),
            "representative_keff": float(nearest["keff"]),
        },
    )


def _rfm_draw_statistics(audit: dict) -> tuple[dict[str, float], int]:
    """The RFM error statistics and how many draws they summarize.

    The wall time is not taken from here: the table's time column comes from the
    device-matched run, in which every method is timed on the same card.
    """
    summary = EXPERIMENT4_RFM_RUN / "rfm_summary.csv"
    if not summary.exists():
        return {}, int(audit.get("rfm_selection", {}).get("draws", 5))
    with summary.open(encoding="utf-8") as handle:
        record = next(r for r in csv.DictReader(handle) if r["method"] == "rfm")
    columns = (
        "keff", "keff_error", "keff_error_min", "keff_error_max",
        "flux_error", "flux_error_min", "flux_error_max",
    )
    return {name: float(record[name]) for name in columns}, int(record["draws"])


def _tex_sci(value: float) -> str:
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.2f}\times 10^{{{exponent}}}"


def _unit_max_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values / max(float(np.max(np.abs(values))), 1.0e-30)


def _load_baseline_flux(
    checkpoint: Path,
    points: np.ndarray,
    subdomains: np.ndarray,
) -> np.ndarray:
    stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = stored["config"]
    model = ResidualFluxNet(int(config["width"]), int(config["outputs"]))
    model.load_state_dict(stored["state_dict"])
    model.eval()
    dtype = next(model.parameters()).dtype
    point_tensor = torch.as_tensor(points, dtype=dtype)
    subdomain_tensor = (
        None
        if int(config["outputs"]) == 1
        else torch.as_tensor(subdomains, dtype=torch.long)
    )
    with torch.no_grad():
        values = model.flux(point_tensor, subdomain_tensor).cpu().numpy()
    return values


def _recompute_baseline_keff(
    checkpoint: Path,
    points: np.ndarray,
    labels: np.ndarray,
    subdomains: np.ndarray,
    robin_points: np.ndarray,
    robin_subdomains: np.ndarray,
) -> float:
    stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = stored["config"]
    model = ResidualFluxNet(int(config["width"]), int(config["outputs"]))
    model.load_state_dict(stored["state_dict"])
    model.eval()
    dtype = next(model.parameters()).dtype
    point_tensor = torch.as_tensor(points, dtype=dtype)
    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    robin_point_tensor = torch.as_tensor(robin_points, dtype=dtype)
    if int(config["outputs"]) == 1:
        subdomain_tensor = None
        robin_subdomain_tensor = None
    else:
        subdomain_tensor = torch.as_tensor(subdomains, dtype=torch.long)
        robin_subdomain_tensor = torch.as_tensor(robin_subdomains, dtype=torch.long)
    # The Rayleigh quotient reads only the interior points, their labels and
    # subdomains, and the Robin trace, so the remaining collocation fields are
    # left empty rather than rebuilt here.
    empty = torch.empty((0, 2), dtype=dtype)
    empty_labels = torch.empty((0,), dtype=torch.long)
    return _evaluate_rayleigh(
        model,
        _TensorCollocation(
            points=point_tensor,
            labels=label_tensor,
            subdomains=subdomain_tensor,
            symmetry_points=empty,
            symmetry_normals=empty,
            symmetry_labels=empty_labels,
            symmetry_subdomains=None,
            robin_points=robin_point_tensor,
            robin_normals=empty,
            robin_labels=empty_labels,
            robin_subdomains=robin_subdomain_tensor,
            interface_points=empty,
            interface_normals=empty,
            interface_left=empty_labels,
            interface_right=empty_labels,
            interface_left_material=empty_labels,
            interface_right_material=empty_labels,
        ),
    )


def generate_experiment4_assets(
    output_root: str | Path, figure_directory: Path | None = None
) -> None:
    output_root = Path(output_root)
    figure_directory = (
        Path(figure_directory) if figure_directory is not None
        else output_root / "figures"
    )
    data_directory = output_root / "data"
    figure_directory.mkdir(parents=True, exist_ok=True)
    data_directory.mkdir(parents=True, exist_ok=True)
    result_root = EXPERIMENT4_RESULTS

    modes, coefficients, summary = _rfm_representative(result_root)
    reference = load_reference(result_root / "reference_subdiv48.npz")
    coordinates = np.linspace(0.0, DOMAIN_SIZE, 171)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel()])
    # The reference and RFM flux fields were once drawn beside the material map
    # and were dropped when those panels were; evaluating them here cost a whole
    # random feature field on every asset build and was then read by nothing.

    material_colors = ListedColormap(["#ffffff", "#c97b63", "#355070", "#6d597a", "#84a59d"])
    material_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], material_colors.N)
    # Only the material map is shown.  The reference and RFM flux fields
    # differ by at most 1.7 steps of a 256-level colormap, so panels of the
    # two are indistinguishable; the comparison they were meant to carry is
    # made quantitatively in the field-error figure.
    fig, axis = plt.subplots(
        1, 1, figsize=(0.42 * TEXT_WIDTH, 2.30), constrained_layout=True
    )
    # A mesh keeps the region boundaries vector; the same map through
    # imshow was a bitmap, and its edges arrived on the page blurred.
    materials = cell_labels()
    edges = np.linspace(0.0, 170.0, materials.shape[0] + 1)
    axis.pcolormesh(
        edges, edges, materials,
        cmap=material_colors, norm=material_norm, rasterized=False,
        edgecolors="face", linewidth=0.0,
    )
    region_labels = (
        (100.0, 100.0, "1"),
        (45.0, 55.0, "2"),
        (5.0, 5.0, "3"),
        (5.0, 80.0, "3"),
        (80.0, 5.0, "3"),
        (80.0, 80.0, "3"),
        (118.0, 118.0, "4"),
    )
    for x_coordinate, y_coordinate, label in region_labels:
        axis.text(
            x_coordinate,
            y_coordinate,
            label,
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            path_effects=[withStroke(linewidth=1.5, foreground="#202020")],
        )
    axis.set_aspect("equal")
    # same reason as the field panels: a grid drawn over the colours is read
    # as part of the map
    axis.grid(False)
    axis.set_xlim(0.0, 170.0)
    axis.set_ylim(0.0, 170.0)
    axis.set_xlabel(r"$x$ (cm)")
    axis.set_ylabel(r"$y$ (cm)")
    fig.savefig(
        figure_directory / "experiment4_rfm_overview.pdf", bbox_inches="tight"
    )
    fig.savefig(
        figure_directory / "experiment4_rfm_overview.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)

    collocation = build_collocation(nx=171, points_per_cell=10)
    evaluation_points = collocation.interior
    evaluation_subdomains = collocation.subdomain
    reference_flux = mean_power_normalize(reference.flux_at(evaluation_points))
    rfm_flux = mean_power_normalize(
        evaluate_random_feature_field(evaluation_points, modes, coefficients)
    )
    acceptance = json.loads(
        (result_root / "baseline_acceptance.json").read_text(encoding="utf-8")
    )
    method_labels = {
        "drm": "DRM",
        "gipmnn": "GIPMNN",
        "pc_gipmnn": "PC-GIPMNN",
        "rfm": "RFM",
    }
    fields: dict[str, np.ndarray] = {}
    checkpoint_keff: dict[str, float] = {}
    selected_seeds: dict[str, int] = {}
    for record in acceptance:
        method = str(record["method"])
        seed = int(record["seed"])
        selected_seeds[method] = seed
        checkpoint = (
            result_root
            / "checkpoints"
            / f"{method}_seed{seed}_selected.pt"
        )
        fields[method] = mean_power_normalize(
            _load_baseline_flux(
                checkpoint, evaluation_points, evaluation_subdomains
            )
        )
        checkpoint_keff[method] = _recompute_baseline_keff(
            checkpoint,
            evaluation_points,
            collocation.material,
            evaluation_subdomains,
            collocation.robin_points,
            collocation.robin_subdomain,
        )
        if not np.isclose(
            checkpoint_keff[method],
            float(record["selected_keff"]),
            rtol=1.0e-7,
            atol=2.0e-8,
        ):
            raise RuntimeError(
                f"{method} checkpoint k_eff {checkpoint_keff[method]} "
                f"does not match {record['selected_keff']}"
            )
    fields["rfm"] = rfm_flux
    reference_max = max(float(np.max(np.abs(reference_flux))), 1.0e-30)
    errors = {
        method: np.abs(values - reference_flux) / reference_max
        for method, values in fields.items()
    }
    common_max = max(float(np.max(values)) for values in errors.values())
    audit = json.loads(
        (result_root / "experiment4_audit.json").read_text(encoding="utf-8")
    )
    rfm_statistics, rfm_draws = _rfm_draw_statistics(audit)
    comparison = {str(row["method"]): dict(row) for row in audit["comparison"]}
    comparison["rfm"].update(rfm_statistics)
    field_rows: list[dict[str, object]] = []
    for method, values in errors.items():
        measured = relative_flux_max_error(fields[method], reference_flux)
        recorded = (
            float(summary["representative_flux_error"])
            if method == "rfm"
            else float(comparison[method]["flux_error"])
        )
        if not np.isclose(measured, recorded, rtol=5.0e-6, atol=2.0e-9):
            raise RuntimeError(
                f"{method} checkpoint field error {measured} does not match {recorded}"
            )
        field_rows.append(
            {
                "method": method,
                "seed": selected_seeds.get(method, int(summary["seed"])),
                "evaluation_points": evaluation_points.shape[0],
                "field_max_error": measured,
                "reported_flux_error": float(comparison[method]["flux_error"]),
                "representative_flux_error": recorded,
                "checkpoint_keff": checkpoint_keff.get(
                    method, float(summary["representative_keff"])
                ),
                "reported_keff": float(comparison[method]["keff"]),
            }
        )
    field_error_by_method = {
        str(row["method"]): float(row["field_max_error"]) for row in field_rows
    }

    # One row, in the order of the summary table, so the panels read as a
    # sequence and the RFM panel closes it.  This is the arrangement of
    # Fig. 12 of Yang et al., against whose field errors these are compared.
    figure, axes = plt.subplots(
        1, 4, figsize=(TEXT_WIDTH, 1.95), constrained_layout=True
    )
    active_grid = labels_at(points) > 0
    for axis, method, panel in zip(
        axes.ravel(),
        ("drm", "gipmnn", "pc_gipmnn", "rfm"),
        ("a", "b", "c", "d"),
        strict=True,
    ):
        grid = np.full(points.shape[0], np.nan)
        grid[active_grid] = errors[method]
        plot = axis.pcolormesh(
            xx,
            yy,
            grid.reshape(xx.shape),
            shading="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=common_max,
            rasterized=True,
        )
        axis.set_title(
            rf"({panel}) {method_labels[method]}" "\n"
            rf"$e_\varphi={_tex_sci(field_error_by_method[method])}$",
            fontsize=7.5,
        )
        axis.set_aspect("equal")
        axis.grid(False)
        axis.set_xlim(0.0, DOMAIN_SIZE)
        axis.set_ylim(0.0, DOMAIN_SIZE)
        axis.set_xlabel(r"$x$ (cm)")
        if method == "drm":
            axis.set_ylabel(r"$y$ (cm)")
        else:
            axis.set_yticklabels([])
    colorbar = figure.colorbar(plot, ax=axes, fraction=0.035, pad=0.02)
    colorbar.set_label(
        r"$|\mathcal{N}\phi-\mathcal{N}\phi_{\rm ref}|/"
        r"\|\mathcal{N}\phi_{\rm ref}\|_{\infty}$",
        fontsize=8,
    )
    colorbar.ax.tick_params(labelsize=7)
    colorbar.ax.yaxis.get_offset_text().set_fontsize(7)
    colorbar.formatter.set_powerlimits((-2, 2))
    colorbar.update_ticks()
    figure.savefig(
        figure_directory / "experiment4_flux_error_comparison.pdf",
        bbox_inches="tight",
    )
    figure.savefig(
        figure_directory / "experiment4_flux_error_comparison.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(figure)

    with (data_directory / "experiment4_field_error_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(field_rows[0]))
        writer.writeheader()
        writer.writerows(field_rows)

    plot_baseline_curves(result_root, figure_directory / "experiment4_baseline_convergence.pdf")

    labels = {
        "drm": "DRM",
        "gipmnn": "GIPMNN",
        "pc_gipmnn": "PC-GIPMNN",
        "rfm": "RFM",
    }
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        rf"\caption{{Errors and computation times of the neural-network baselines and the RFM for the IAEA quarter-core benchmark. RFM entries are medians over {_spelled(rfm_draws)} independent feature draws, with the observed range in brackets.}}",
        r"\label{tab:experiment4_iea_neutron}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"method & $k_{\rm eff}$ & $e_k$ & $e_\varphi$ & time (s)\\",
        r"\midrule",
    ]
    # Wall times come from the device-matched run, in which the RFM solves on
    # the same card the baselines were trained on.  Without this the column
    # would hold a host time next to three device times.
    timing_file = result_root / "device_matched_timing.json"
    timings = json.loads(timing_file.read_text(encoding="utf-8"))["methods"]
    csv_rows: list[dict[str, object]] = []
    for row in audit["comparison"]:
        method = str(row["method"])
        row = dict(row)
        if method == "rfm":
            row.update(rfm_statistics)
        entry = timings[method]
        row["seconds"] = float(entry["seconds"])
        row["timing_device"] = str(entry["device"])
        row["timing_dtype"] = str(entry["dtype"])
        if method == "rfm":
            # A nested tabular rather than \shortstack, which sets its baseline
            # at the foot of the stack: the median then climbed into the row
            # above and read as the preceding method's.  The [t] alignment
            # keeps it on its own row and hangs the range below.
            keff_error = (
                r"\begin{tabular}[t]{@{}c@{}}"
                rf"\({_tex_sci(float(row['keff_error']))}\)\\"
                rf"\([{_tex_sci(float(row['keff_error_min']))},"
                rf"{_tex_sci(float(row['keff_error_max']))}]\)"
                r"\end{tabular}"
            )
            flux_error = (
                r"\begin{tabular}[t]{@{}c@{}}"
                rf"\({_tex_sci(float(row['flux_error']))}\)\\"
                rf"\([{_tex_sci(float(row['flux_error_min']))},"
                rf"{_tex_sci(float(row['flux_error_max']))}]\)"
                r"\end{tabular}"
            )
        else:
            keff_error = rf"\({_tex_sci(float(row['keff_error']))}\)"
            flux_error = rf"\({_tex_sci(float(row['flux_error']))}\)"
        lines.append(
            f"{labels[method]} & {float(row['keff']):.8f} & "
            f"{keff_error} & {flux_error} & {float(row['seconds']):.1f}\\\\"
        )
        csv_rows.append(dict(row))
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (data_directory / "experiment4_rewrite_summary_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    with (data_directory / "experiment4_reproduced_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
