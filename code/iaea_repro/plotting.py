from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_baseline_curves(result_root: Path, output: Path) -> None:
    methods = (
        ("drm_seed2026071101", "DRM", "#4477AA"),
        ("gipmnn_seed2026071102", "GIPMNN", "#CC6677"),
        ("pc_gipmnn_seed2026071103", "PC-GIPMNN", "#228833"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    from .reference import load_reference

    reference = load_reference(result_root / "reference_subdiv48.npz")
    found = 0
    for directory, label, color in methods:
        path = result_root / directory / "convergence.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        epoch = np.asarray([int(row["epoch"]) for row in rows])
        keff_values = np.asarray([float(row["keff"]) for row in rows])
        keff = np.abs(keff_values - reference.keff) / abs(reference.keff)
        flux = np.asarray([float(row["flux_error"]) for row in rows])
        axes[0].loglog(epoch, keff, color=color, label=label, linewidth=1.05)
        axes[1].loglog(epoch, flux, color=color, label=label, linewidth=1.05)
        summary_path = result_root / directory / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            selected_epoch = int(summary["selected_epoch"])
            selected_index = int(np.argmin(np.abs(epoch - selected_epoch)))
            axes[0].plot(epoch[selected_index], keff[selected_index], "o", color=color, ms=4.2)
            axes[1].plot(epoch[selected_index], flux[selected_index], "o", color=color, ms=4.2)
        found += 1
    if not found:
        raise FileNotFoundError("No baseline convergence files were found.")
    for axis in axes:
        axis.axvline(5.0e4, color="#666666", linestyle="--", linewidth=0.8)
        axis.axvline(5.0e5, color="#999999", linestyle="-.", linewidth=0.8)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel(r"relative error in $k_{\mathrm{eff}}$")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel(r"relative max error in $\phi$")
    for axis in axes:
        axis.grid(True, which="both", linestyle=":", linewidth=0.5)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
