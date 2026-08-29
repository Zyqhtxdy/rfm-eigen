from __future__ import annotations

import csv
import hashlib
import json
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from paper_assets.paths import EXPERIMENT4_RESULTS

from .geometry import (
    CELL_SIZE,
    DOMAIN_SIZE,
    MATERIALS,
    build_collocation,
    cell_labels,
    geometry_sha256,
)
from .metrics import relative_flux_max_error
from .reference import load_reference


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RFMSolution:
    seed: int
    features: int
    rank: int
    cutoff: float
    frequency_radius: float
    keff: float
    keff_error: float
    flux_error: float
    seconds: float
    coefficients: np.ndarray
    modes: np.ndarray
    phases: np.ndarray
    energy_values: np.ndarray
    ritz_values: np.ndarray
    field_points: np.ndarray
    field: np.ndarray
    algebraic_residual: float
    timings: dict[str, float]


def sample_random_cosine_modes(count: int, seed: int, radius: float) -> np.ndarray:
    """Return a constant mode followed by iid frequencies from a triangle."""
    if count < 2:
        raise ValueError("At least two features are required.")
    if radius <= 0.0:
        raise ValueError("The frequency radius must be positive.")
    rng = np.random.default_rng(seed)
    unit = rng.random((count - 1, 2))
    reflected = unit.sum(axis=1) > 1.0
    unit[reflected] = 1.0 - unit[reflected]
    return np.vstack((np.zeros((1, 2)), radius * unit))


def _cos_integral(frequency: np.ndarray, phase: np.ndarray, left: float, right: float) -> np.ndarray:
    output = np.empty_like(frequency)
    nonzero = np.abs(frequency) > 1.0e-14
    output[nonzero] = (
        np.sin(frequency[nonzero] * right + phase[nonzero])
        - np.sin(frequency[nonzero] * left + phase[nonzero])
    ) / frequency[nonzero]
    output[~nonzero] = (right - left) * np.cos(phase[~nonzero])
    return output


def _pair_integrals(
    frequencies: np.ndarray, phases: np.ndarray, left: float, right: float
) -> tuple[np.ndarray, np.ndarray]:
    a = frequencies[:, None]
    b = frequencies[None, :]
    p = phases[:, None]
    q = phases[None, :]
    difference = _cos_integral(a - b, p - q, left, right)
    total = _cos_integral(a + b, p + q, left, right)
    return 0.5 * (difference + total), 0.5 * (difference - total)


def _axis_integrals(
    frequencies: np.ndarray, phases: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    cosine: list[np.ndarray] = []
    sine: list[np.ndarray] = []
    for index in range(17):
        left = CELL_SIZE * index
        c_value, s_value = _pair_integrals(frequencies, phases, left, left + CELL_SIZE)
        cosine.append(c_value)
        sine.append(s_value)
    return cosine, sine


def _boundary_edges() -> list[tuple[int, int, int, int]]:
    labels = cell_labels()
    edges: list[tuple[int, int, int, int]] = []
    for iy in range(17):
        for ix in range(17):
            if labels[iy, ix] <= 0:
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                jx, jy = ix + dx, iy + dy
                active = 0 <= jx < 17 and 0 <= jy < 17 and labels[jy, jx] > 0
                symmetry = ix == 0 and dx == -1 or iy == 0 and dy == -1
                if not active and not symmetry:
                    edges.append((ix, iy, dx, dy))
    return edges


def assemble_random_feature_matrices(modes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = modes.shape[0]
    alpha = np.pi * modes[:, 0] / DOMAIN_SIZE
    beta = np.pi * modes[:, 1] / DOMAIN_SIZE
    phase = np.zeros(count)
    cos_x, sin_x = _axis_integrals(alpha, phase)
    cos_y, sin_y = _axis_integrals(beta, phase)
    alpha_pair = alpha[:, None] * alpha[None, :]
    beta_pair = beta[:, None] * beta[None, :]
    stiffness = np.zeros((count, count), dtype=float)
    fission = np.zeros_like(stiffness)
    labels = cell_labels()

    for iy, ix in np.argwhere(labels > 0):
        material = MATERIALS[int(labels[iy, ix])]
        mass_cell = cos_x[ix] * cos_y[iy]
        stiffness += material.diffusion * (
            alpha_pair * sin_x[ix] * cos_y[iy]
            + beta_pair * cos_x[ix] * sin_y[iy]
        )
        stiffness += material.sigma_a * mass_cell
        fission += material.nu_sigma_f * mass_cell

    for ix, iy, dx, dy in _boundary_edges():
        if dx:
            coordinate = CELL_SIZE * (ix + (dx > 0))
            trace = np.cos(alpha * coordinate)
            stiffness += 0.5 * trace[:, None] * trace[None, :] * cos_y[iy]
        else:
            coordinate = CELL_SIZE * (iy + (dy > 0))
            trace = np.cos(beta * coordinate)
            stiffness += 0.5 * trace[:, None] * trace[None, :] * cos_x[ix]
    return 0.5 * (stiffness + stiffness.T), 0.5 * (fission + fission.T)


def evaluate_random_feature_field(
    points: np.ndarray, modes: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    values = np.empty(points.shape[0], dtype=float)
    alpha = np.pi * modes[:, 0] / DOMAIN_SIZE
    beta = np.pi * modes[:, 1] / DOMAIN_SIZE
    for start in range(0, points.shape[0], 4096):
        stop = min(points.shape[0], start + 4096)
        x_part = np.cos(points[start:stop, 0, None] * alpha[None, :])
        y_part = np.cos(points[start:stop, 1, None] * beta[None, :])
        values[start:stop] = (x_part * y_part) @ coefficients
    if float(values.mean()) < 0.0:
        values = -values
    return values / max(float(np.max(np.abs(values))), 1.0e-30)


def solve_random_feature_rfm(
    features: int,
    cutoff: float,
    seed: int,
    frequency_radius: float,
    reference_subdiv: int = 48,
    *,
    evaluate_reference: bool = True,
) -> RFMSolution:
    stages: dict[str, float] = {}
    started = time.perf_counter()
    stage_started = time.perf_counter()
    modes = sample_random_cosine_modes(features, seed, frequency_radius)
    phases = np.zeros_like(modes)
    stages["sample_features_seconds"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    stiffness, fission = assemble_random_feature_matrices(modes)
    stages["assembly_seconds"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    energy_values, energy_vectors = eigh(stiffness, check_finite=False)
    threshold = max(float(energy_values[-1]) * cutoff, 1.0e-15)
    retained = energy_values > threshold
    whitening = energy_vectors[:, retained] / np.sqrt(energy_values[retained])[None, :]
    stages["whitening_seconds"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    reduced = whitening.T @ fission @ whitening
    index = int(retained.sum()) - 1
    keff_values, vectors = eigh(reduced, subset_by_index=[index, index], check_finite=False)
    coefficients = whitening @ vectors[:, 0]
    keff = float(keff_values[0])
    ritz_values = np.asarray(keff_values, dtype=float)
    fission_applied = fission @ coefficients
    stiffness_applied = stiffness @ coefficients
    algebraic_residual = float(
        np.linalg.norm(fission_applied - keff * stiffness_applied)
        / max(
            np.linalg.norm(fission_applied) + abs(keff) * np.linalg.norm(stiffness_applied),
            1.0e-30,
        )
    )
    stages["eigensolve_seconds"] = time.perf_counter() - stage_started
    production_seconds = time.perf_counter() - started

    stage_started = time.perf_counter()
    points = build_collocation().interior
    field = evaluate_random_feature_field(points, modes, coefficients)
    stages["field_evaluation_seconds"] = time.perf_counter() - stage_started
    keff_error = float("nan")
    flux_error = float("nan")
    if evaluate_reference:
        stage_started = time.perf_counter()
        reference = load_reference(
            EXPERIMENT4_RESULTS / f"reference_subdiv{reference_subdiv}.npz"
        )
        reference_flux = reference.flux_at(points)
        keff_error = abs(keff - reference.keff) / abs(reference.keff)
        flux_error = relative_flux_max_error(field, reference_flux)
        stages["post_lock_reference_acceptance_seconds"] = (
            time.perf_counter() - stage_started
        )
    return RFMSolution(
        seed=seed,
        features=features,
        rank=int(retained.sum()),
        cutoff=cutoff,
        frequency_radius=frequency_radius,
        keff=keff,
        keff_error=keff_error,
        flux_error=flux_error,
        seconds=production_seconds,
        coefficients=coefficients,
        modes=modes,
        phases=phases,
        energy_values=energy_values,
        ritz_values=ritz_values,
        field_points=points,
        field=field,
        algebraic_residual=algebraic_residual,
        timings=stages,
    )


def save_rfm_solution_artifact(
    solution: RFMSolution,
    path: str | Path,
    *,
    device_provenance: dict[str, str] | None = None,
) -> Path:
    """Save every stochastic and numerical state needed to audit one draw."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    timing_names = np.asarray(list(solution.timings), dtype="U64")
    timing_seconds = np.asarray([solution.timings[key] for key in timing_names], dtype=float)
    provenance = device_provenance or {
        "numerical_core_device": "unspecified",
        "feature_sampling_device": "unspecified",
        "field_evaluation_device": "unspecified",
        "artifact_serialization_device": "cpu",
    }
    np.savez_compressed(
        output,
        seed=solution.seed,
        features=solution.features,
        retained_rank=solution.rank,
        energy_cutoff=solution.cutoff,
        frequency_radius=solution.frequency_radius,
        feature_modes=solution.modes,
        feature_phases=solution.phases,
        coefficients=solution.coefficients,
        energy_values=solution.energy_values,
        ritz_values=solution.ritz_values,
        field_points=solution.field_points,
        normalized_field=solution.field,
        keff=solution.keff,
        generalized_eigen_residual=solution.algebraic_residual,
        production_seconds=solution.seconds,
        timing_names=timing_names,
        timing_seconds=timing_seconds,
        reference_used_for_production=False,
        numerical_core_device=np.asarray(provenance["numerical_core_device"]),
        feature_sampling_device=np.asarray(provenance["feature_sampling_device"]),
        field_evaluation_device=np.asarray(provenance["field_evaluation_device"]),
        artifact_serialization_device=np.asarray(
            provenance.get("artifact_serialization_device", "cpu")
        ),
    )
    return output


def evaluate_locked_rfm_acceptance(
    solution: RFMSolution,
    reference_subdiv: int,
) -> dict[str, float | int | str]:
    """Evaluate a previously locked draw once against the independent FV field."""
    started = time.perf_counter()
    reference_path = EXPERIMENT4_RESULTS / f"reference_subdiv{reference_subdiv}.npz"
    reference = load_reference(reference_path)
    reference_flux = reference.flux_at(solution.field_points)
    return {
        "reference_access_stage": "post_lock_acceptance_only",
        "reference_evaluations": 1,
        "reference_subdiv": reference_subdiv,
        "reference_path": str(reference_path),
        "reference_keff": reference.keff,
        "keff_error": abs(solution.keff - reference.keff) / abs(reference.keff),
        "flux_error": relative_flux_max_error(solution.field, reference_flux),
        "acceptance_seconds": time.perf_counter() - started,
    }


def search_random_feature_rfm(
    result_directory: Path,
    feature_counts: list[int],
    frequency_radii: list[float],
    cutoffs: list[float],
    seed: int,
    selection_relative_tolerance: float = 5.0e-4,
    selection_rule: str = (
        "least-cost configuration within tolerance of the minimum "
        "reference-free generalized-eigen residual"
    ),
) -> list[dict[str, float | int]]:
    """Exploratory search using only an equation-algebra residual.

    This routine is deliberately incapable of receiving a reference mesh,
    target error, published k-effective value, or reference flux.  Formal
    reporting uses the separately preregistered final_* fields in rfm.toml.
    """
    result_directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int]] = []
    solutions: list[RFMSolution] = []
    for features in feature_counts:
        for frequency_radius in frequency_radii:
            for cutoff in cutoffs:
                solution = solve_random_feature_rfm(
                    features,
                    cutoff,
                    seed,
                    frequency_radius,
                    evaluate_reference=False,
                )
                row = {
                    "seed": solution.seed,
                    "features": solution.features,
                    "rank": solution.rank,
                    "frequency_radius": solution.frequency_radius,
                    "cutoff": solution.cutoff,
                    "keff": solution.keff,
                    "generalized_eigen_residual": solution.algebraic_residual,
                    "selection_uses_reference": 0,
                    "seconds": solution.seconds,
                }
                rows.append(row)
                solutions.append(solution)
                save_rfm_solution_artifact(
                    solution,
                    result_directory
                    / (
                        f"draw_N{features}_R{frequency_radius:g}_"
                        f"cutoff{cutoff:.1e}_seed{seed}.npz"
                    ),
                )
                print(json.dumps(row), flush=True)
    search_path = result_directory / "search.csv"
    if search_path.exists():
        raise FileExistsError(f"Refusing to overwrite exploratory search {search_path}")
    with search_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if solutions:
        best_index = int(
            np.argmin([float(row["generalized_eigen_residual"]) for row in rows])
        )
        best_score = float(rows[best_index]["generalized_eigen_residual"])
        eligible = [
            (solution, row)
            for solution, row in zip(solutions, rows)
            if float(row["generalized_eigen_residual"])
            <= best_score * (1.0 + selection_relative_tolerance)
        ]
        selected, selected_row = min(
            eligible,
            key=lambda item: (
                item[0].features,
                item[0].rank,
                item[0].frequency_radius,
                -item[0].cutoff,
            ),
        )
        save_rfm_solution_artifact(selected, result_directory / "selected_solution.npz")
        selection = {
            "selection_rule": selection_rule,
            "selection_relative_tolerance": selection_relative_tolerance,
            "selection_metric": "relative_generalized_eigen_residual",
            "reference_assisted": False,
            "reference_inputs_accepted_by_search_api": False,
            "best": dict(rows[best_index]),
            "selected": dict(selected_row),
        }
        with (result_directory / "selection.json").open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(selection, handle, indent=2)
    return rows


def search_configured_rfm(config_path: str | Path) -> dict[str, object]:
    with Path(config_path).open("rb") as handle:
        config = tomllib.load(handle)
    result_directory = EXPERIMENT4_RESULTS / str(config["search_result_name"])
    rows = search_random_feature_rfm(
        result_directory=result_directory,
        feature_counts=[int(value) for value in config["feature_counts"]],
        frequency_radii=[float(value) for value in config["frequency_radii"]],
        cutoffs=[float(value) for value in config["energy_cutoffs"]],
        seed=int(config["seed"]),
        selection_relative_tolerance=float(config["selection_relative_tolerance"]),
        selection_rule=str(config["selection_rule"]),
    )
    selection = json.loads((result_directory / "selection.json").read_text(encoding="utf-8"))
    return {"rows": len(rows), "result_directory": str(result_directory), **selection}


def run_configured_rfm(
    config_path: str | Path,
    *,
    output_directory: str | Path | None = None,
) -> dict[str, object]:
    with Path(config_path).open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("configuration_status") == (
        "historical_reference_informed_locked_for_confirmatory_new_seeds"
    ):
        raise RuntimeError(
            "The historical-parameter confirmatory protocol is same-GPU only; "
            "use experiments/section4_fair/run_exp4_fair_retrain.py.  The CPU "
            "solver in run_configured_rfm is not admissible beside CUDA neural training."
        )
    if config.get("configuration_status") != "preregistered_before_reference_evaluation":
        raise ValueError(
            "Formal Example 4 RFM runs require a preregistered configuration status."
        )
    if bool(config.get("selection_uses_reference", True)):
        raise ValueError("Formal RFM hyperparameter selection must be reference-free")
    run_directory = (
        Path(output_directory)
        if output_directory is not None
        else EXPERIMENT4_RESULTS / "fair_retrains" / str(config["result_name"])
    )
    if run_directory.exists() and any(run_directory.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite formal RFM run {run_directory}; choose a new run_id."
        )
    run_directory.mkdir(parents=True, exist_ok=True)
    reporting_seeds = [int(value) for value in config["reporting_seeds"]]
    representative_seed = int(config["representative_seed"])
    if representative_seed not in reporting_seeds:
        raise ValueError("representative_seed must be preregistered among reporting_seeds")

    locked_protocol = {
        "configuration_status": config["configuration_status"],
        "selection_uses_reference": False,
        "selection_rule": str(config["selection_rule"]),
        "final_features": int(config["final_features"]),
        "final_frequency_radius": float(config["final_frequency_radius"]),
        "final_energy_cutoff": float(config["final_energy_cutoff"]),
        "reporting_seeds": reporting_seeds,
        "representative_seed": representative_seed,
        "reference_subdiv_reserved_for_post_lock_acceptance": int(
            config["reference_subdiv"]
        ),
        "geometry_sha256": geometry_sha256(),
    }
    protocol_path = run_directory / "locked_protocol.json"
    with protocol_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(locked_protocol, handle, indent=2)
    protocol_sha256 = _file_sha256(protocol_path)

    solutions: list[RFMSolution] = []
    acceptances: list[dict[str, object]] = []
    for seed in reporting_seeds:
        solution = solve_random_feature_rfm(
            int(config["final_features"]),
            float(config["final_energy_cutoff"]),
            seed,
            float(config["final_frequency_radius"]),
            evaluate_reference=False,
        )
        artifact_path = save_rfm_solution_artifact(
            solution, run_directory / f"solution_seed{seed}.npz"
        )
        artifact_sha256 = _file_sha256(artifact_path)
        acceptance: dict[str, object] = {
            **evaluate_locked_rfm_acceptance(
                solution, int(config["reference_subdiv"])
            ),
            "seed": seed,
            "locked_artifact": str(artifact_path),
            "locked_artifact_sha256": artifact_sha256,
            "locked_protocol_sha256": protocol_sha256,
        }
        with (run_directory / f"acceptance_seed{seed}.json").open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(acceptance, handle, indent=2)
        solutions.append(solution)
        acceptances.append(acceptance)

    keff_values = np.asarray([solution.keff for solution in solutions])
    keff_errors = np.asarray([float(item["keff_error"]) for item in acceptances])
    flux_errors = np.asarray([float(item["flux_error"]) for item in acceptances])
    ranks = np.asarray([solution.rank for solution in solutions])
    seconds = np.asarray([solution.seconds for solution in solutions])
    residuals = np.asarray([solution.algebraic_residual for solution in solutions])
    representative_index = reporting_seeds.index(representative_seed)
    representative = solutions[representative_index]
    representative_acceptance = acceptances[representative_index]
    summary: dict[str, object] = {
        "method": "Random feature method (RFM)",
        "seed": representative_seed,
        "representative_seed": representative_seed,
        "representative_rule": "preregistered seed; no error-based representative selection",
        "search_seed": int(config["seed"]),
        "reporting_seeds": reporting_seeds,
        "reporting_draws": len(solutions),
        "feature_law": "constant mode plus iid uniform triangular cosine frequencies",
        "features": representative.features,
        "retained_rank": int(np.median(ranks)),
        "retained_rank_min": int(np.min(ranks)),
        "retained_rank_max": int(np.max(ranks)),
        "frequency_radius": representative.frequency_radius,
        "energy_cutoff": representative.cutoff,
        "keff": float(np.median(keff_values)),
        "keff_error": float(np.median(keff_errors)),
        "keff_error_min": float(np.min(keff_errors)),
        "keff_error_max": float(np.max(keff_errors)),
        "flux_error": float(np.median(flux_errors)),
        "flux_error_min": float(np.min(flux_errors)),
        "flux_error_max": float(np.max(flux_errors)),
        "representative_keff": representative.keff,
        "representative_keff_error": float(representative_acceptance["keff_error"]),
        "representative_flux_error": float(representative_acceptance["flux_error"]),
        "flux_error_definition": "Yang et al. (2023), Eqs. (37) and (42)",
        "seconds": float(np.median(seconds)),
        "seconds_min": float(np.min(seconds)),
        "seconds_max": float(np.max(seconds)),
        "generalized_eigen_residual": float(np.median(residuals)),
        "generalized_eigen_residual_min": float(np.min(residuals)),
        "generalized_eigen_residual_max": float(np.max(residuals)),
        "reference_subdiv": int(config["reference_subdiv"]),
        "reference_access_stage": "post_lock_acceptance_only",
        "reference_evaluations_per_draw": 1,
        "configuration_status": config["configuration_status"],
        "selection_reference_assisted": False,
        "selection_metric": str(config["selection_metric"]),
        "selection_rule": str(config["selection_rule"]),
        "search_result_name": str(config["search_result_name"]),
        "locked_protocol_sha256": protocol_sha256,
        "geometry_sha256": geometry_sha256(),
    }
    trial_rows = []
    for solution in solutions:
        row = {
            "seed": solution.seed,
            "features": solution.features,
            "retained_rank": solution.rank,
            "frequency_radius": solution.frequency_radius,
            "energy_cutoff": solution.cutoff,
            "keff": solution.keff,
            "keff_error": float(acceptances[len(trial_rows)]["keff_error"]),
            "flux_error": float(acceptances[len(trial_rows)]["flux_error"]),
            "generalized_eigen_residual": solution.algebraic_residual,
            "seconds": solution.seconds,
            **solution.timings,
        }
        trial_rows.append(row)
    with (run_directory / "trials.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trial_rows[0]))
        writer.writeheader()
        writer.writerows(trial_rows)
    with (run_directory / "summary.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(summary, handle, indent=2)
    return summary


def audit_rfm_results(
    result_name: str = "rfm_preregistered_random_fourier",
) -> dict[str, object]:
    """Verify locks and artifact completeness without reusing errors for selection."""
    run_directory = EXPERIMENT4_RESULTS / "fair_retrains" / result_name
    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    protocol_path = run_directory / "locked_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if bool(protocol["selection_uses_reference"]):
        raise RuntimeError("Locked protocol unexpectedly reports reference-assisted selection")
    if _file_sha256(protocol_path) != summary["locked_protocol_sha256"]:
        raise RuntimeError("Locked RFM protocol hash mismatch")
    required_arrays = {
        "feature_modes",
        "feature_phases",
        "coefficients",
        "energy_values",
        "ritz_values",
        "field_points",
        "normalized_field",
        "timing_names",
        "timing_seconds",
    }
    draws: list[dict[str, object]] = []
    for seed in protocol["reporting_seeds"]:
        artifact_path = run_directory / f"solution_seed{seed}.npz"
        acceptance = json.loads(
            (run_directory / f"acceptance_seed{seed}.json").read_text(encoding="utf-8")
        )
        if _file_sha256(artifact_path) != acceptance["locked_artifact_sha256"]:
            raise RuntimeError(f"Locked artifact hash mismatch for seed {seed}")
        with np.load(artifact_path, allow_pickle=False) as payload:
            missing = sorted(required_arrays.difference(payload.files))
            if missing:
                raise RuntimeError(f"Seed {seed} artifact is missing {missing}")
            if bool(payload["reference_used_for_production"]):
                raise RuntimeError(f"Seed {seed} production artifact used a reference")
        draws.append(
            {
                "seed": seed,
                "artifact_sha256": acceptance["locked_artifact_sha256"],
                "reference_evaluations": acceptance["reference_evaluations"],
            }
        )
    return {
        "selection_reference_assisted": False,
        "locked_protocol_sha256": summary["locked_protocol_sha256"],
        "artifact_contract_complete": True,
        "draws": draws,
    }
