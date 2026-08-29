"""Offline Example 4 reference refinement and acceptance.

Training and RFM production never import a stored reference.  This module is
entered only after the wrapper has hashed every neural and RFM lock.  It then
builds fresh FV references at subdivisions 48, 64, and 96 inside the immutable
run directory and evaluates every locked field against the same final field.
"""
from __future__ import annotations

import hashlib
import json
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import RunConfig
from .geometry import build_collocation, geometry_sha256
from .metrics import relative_flux_max_error, relative_keff_error
from .reference import load_reference, save_reference, solve_reference


@dataclass(frozen=True)
class OfflineAcceptanceConfig:
    method: str
    reference_subdiv: int
    refinement_subdivisions: tuple[int, ...]
    published_keff: float
    target_keff_error: float
    target_flux_error: float
    source: str


def load_offline_acceptance(
    path: str | Path,
    method: str,
) -> OfflineAcceptanceConfig:
    with Path(path).open("rb") as handle:
        values = tomllib.load(handle)
    if method not in values:
        raise ValueError(f"offline acceptance has no method table {method!r}")
    method_values = values[method]
    subdivisions = tuple(
        int(value) for value in values.get("refinement_subdivisions", [48, 64, 96])
    )
    if subdivisions != tuple(sorted(set(subdivisions))) or len(subdivisions) < 3:
        raise ValueError("reference refinement requires at least three increasing grids")
    if int(values["reference_subdiv"]) != subdivisions[-1]:
        raise ValueError("acceptance reference_subdiv must be the finest refinement grid")
    return OfflineAcceptanceConfig(
        method=method,
        reference_subdiv=int(values["reference_subdiv"]),
        refinement_subdivisions=subdivisions,
        published_keff=float(values["published_keff"]),
        target_keff_error=float(method_values["target_keff_error"]),
        target_flux_error=float(method_values["target_flux_error"]),
        source=str(method_values["source"]),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)


def build_fresh_reference_refinement(
    output_directory: str | Path,
    *,
    subdivisions: tuple[int, ...] = (48, 64, 96),
    keff_relative_tolerance: float = 1.0e-5,
    flux_relative_max_tolerance: float = 1.0e-2,
    resume: bool = False,
) -> dict[str, object]:
    """Build and lock a fresh FV refinement sequence in the current run."""
    if subdivisions != tuple(sorted(set(subdivisions))) or len(subdivisions) < 3:
        raise ValueError("subdivisions must contain at least three increasing values")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "reference_refinement.json"
    if resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if tuple(manifest["subdivisions"]) != subdivisions:
            raise RuntimeError("reference-refinement grids changed on resume")
        for row in manifest["references"]:
            path = Path(row["path"])
            if _sha256(path) != row["sha256"]:
                raise RuntimeError("fresh reference artifact changed before resume")
        return manifest
    if manifest_path.exists() or (
        not resume and output.exists() and any(output.iterdir())
    ):
        raise FileExistsError(
            f"fresh reference directory must be empty unless --resume is used: {output}"
        )

    points = build_collocation(nx=171, points_per_cell=10).interior
    references: list[dict[str, object]] = []
    fields: list[np.ndarray] = []
    previous_keff: float | None = None
    previous_field: np.ndarray | None = None
    for subdiv in subdivisions:
        path = output / f"reference_subdiv{subdiv}.npz"
        sidecar = path.with_suffix(".production.json")
        if resume and path.exists():
            solution = load_reference(path)
            if solution.subdiv != subdiv:
                raise RuntimeError("orphan reference subdivision differs from protocol")
            if sidecar.exists():
                production = json.loads(sidecar.read_text(encoding="utf-8"))
                if production["sha256"] != _sha256(path):
                    raise RuntimeError("orphan reference sidecar hash mismatch")
                solve_seconds = production["solve_seconds"]
                recovered = bool(production.get("recovered_after_interrupted_finalization", False))
            else:
                solve_seconds = None
                recovered = True
                _write_json_exclusive(
                    sidecar,
                    {
                        "subdiv": subdiv,
                        "path": str(path.resolve()),
                        "sha256": _sha256(path),
                        "keff": solution.keff,
                        "solve_seconds": None,
                        "recovered_after_interrupted_finalization": True,
                        "unrecoverable_timing_fields_are_null": True,
                    },
                )
        elif sidecar.exists():
            raise RuntimeError("orphan reference sidecar has no reference artifact")
        else:
            started = time.perf_counter()
            solution = solve_reference(subdiv=subdiv, tolerance=1.0e-10)
            solve_seconds = time.perf_counter() - started
            save_reference(solution, path)
            recovered = False
            _write_json_exclusive(
                sidecar,
                {
                    "subdiv": subdiv,
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "keff": solution.keff,
                    "solve_seconds": solve_seconds,
                    "recovered_after_interrupted_finalization": False,
                },
            )
        field = solution.flux_at(points)
        keff_change = (
            None
            if previous_keff is None
            else abs(solution.keff - previous_keff) / abs(solution.keff)
        )
        flux_change = (
            None
            if previous_field is None
            else relative_flux_max_error(field, previous_field)
        )
        references.append(
            {
                "subdiv": subdiv,
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "keff": solution.keff,
                "relative_keff_change_from_previous": keff_change,
                "relative_flux_max_change_from_previous": flux_change,
                "solve_seconds": solve_seconds,
                "recovered_after_interrupted_finalization": recovered,
                "freshly_solved_in_current_run": True,
                "old_reference_artifact_read": False,
            }
        )
        fields.append(field)
        previous_keff = solution.keff
        previous_field = field

    last = references[-1]
    refinement_converged = bool(
        float(last["relative_keff_change_from_previous"]) <= keff_relative_tolerance
        and float(last["relative_flux_max_change_from_previous"])
        <= flux_relative_max_tolerance
    )
    manifest = {
        "subdivisions": list(subdivisions),
        "geometry_sha256": geometry_sha256(),
        "evaluation_points": int(points.shape[0]),
        "reference_generation": "fresh_finite_volume_solve_in_current_run",
        "old_reference_artifact_read": False,
        "references": references,
        "final_reference_path": references[-1]["path"],
        "final_reference_sha256": references[-1]["sha256"],
        "keff_relative_tolerance": keff_relative_tolerance,
        "flux_relative_max_tolerance": flux_relative_max_tolerance,
        "refinement_converged": refinement_converged,
    }
    _write_json_exclusive(manifest_path, manifest)
    if not refinement_converged:
        raise RuntimeError(
            "fresh FV 48/64/96 refinement did not converge; artifacts were preserved"
        )
    return manifest


def _validated_reference(
    acceptance: OfflineAcceptanceConfig,
    refinement_manifest_path: str | Path,
):
    manifest_path = Path(refinement_manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest["refinement_converged"]:
        raise RuntimeError("unconverged FV refinement cannot be used for acceptance")
    if int(manifest["subdivisions"][-1]) != acceptance.reference_subdiv:
        raise RuntimeError("offline acceptance and refinement final grid differ")
    reference_path = Path(manifest["final_reference_path"])
    if _sha256(reference_path) != manifest["final_reference_sha256"]:
        raise RuntimeError("final FV reference hash changed before acceptance")
    return manifest, reference_path, load_reference(reference_path)


def evaluate_locked_baseline(
    run_directory: str | Path,
    acceptance: OfflineAcceptanceConfig,
    refinement_manifest_path: str | Path,
    *,
    resume: bool = False,
) -> dict[str, object]:
    """Evaluate a locked neural field using the fresh common FV reference."""
    run_dir = Path(run_directory)
    acceptance_path = run_dir / "acceptance.json"
    if acceptance_path.exists():
        if not resume:
            raise FileExistsError(f"offline acceptance already exists: {acceptance_path}")
        return json.loads(acceptance_path.read_text(encoding="utf-8"))
    training_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    training_config_values = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    forbidden = {
        "reference_subdiv",
        "published_keff",
        "target_keff_error",
        "target_flux_error",
    }
    leaked = sorted(forbidden.intersection(training_config_values))
    if leaked:
        raise RuntimeError(f"training configuration leaked offline fields: {leaked}")
    config = RunConfig(**training_config_values)
    config.validate()
    if config.method != acceptance.method:
        raise ValueError("offline method does not match the locked training method")

    locked_path = run_dir / "locked_model.pt"
    locked_sha256 = _sha256(locked_path)
    if locked_sha256 != training_summary["locked_checkpoint_sha256"]:
        raise RuntimeError("locked baseline checkpoint hash changed before acceptance")
    field_path = run_dir / "locked_field.npz"
    field_sha256 = _sha256(field_path)
    if field_sha256 != training_summary["locked_field_sha256"]:
        raise RuntimeError("locked baseline field hash changed before acceptance")
    with np.load(field_path, allow_pickle=False) as payload:
        if bool(payload["reference_used"]):
            raise RuntimeError("locked baseline field was not reference-free")
        points = np.asarray(payload["points"], dtype=float)
        approximation = np.asarray(payload["normalized_field"], dtype=float)
        keff = float(payload["keff"])

    started = time.perf_counter()
    refinement, reference_path, reference = _validated_reference(
        acceptance, refinement_manifest_path
    )
    reference_flux = reference.flux_at(points)
    keff_error = relative_keff_error(keff, acceptance.published_keff)
    flux_error = relative_flux_max_error(approximation, reference_flux)
    accuracy_accepted = bool(
        keff_error <= acceptance.target_keff_error
        and flux_error <= acceptance.target_flux_error
    )
    source_budget_complete = bool(training_summary["source_budget_checkpoint_complete"])
    reference_free_converged = bool(training_summary["reference_free_converged"])
    source_accuracy_accepted = bool(source_budget_complete and accuracy_accepted)
    extended_convergence_gate_passed = bool(
        config.training_track == "extended_to_convergence"
        and reference_free_converged
    )
    record = {
        "run_id": training_summary["run_id"],
        "method": config.method,
        "seed": config.seed,
        "training_track": config.training_track,
        "reference_access_stage": "offline_after_all_nn_and_rfm_locks",
        "reference_evaluations": 1,
        "offline_config_source": acceptance.source,
        "locked_checkpoint": str(locked_path),
        "locked_checkpoint_sha256": locked_sha256,
        "locked_field": str(field_path),
        "locked_field_sha256": field_sha256,
        "reference_refinement_manifest": str(Path(refinement_manifest_path)),
        "reference_refinement_converged": refinement["refinement_converged"],
        "reference_subdiv": reference.subdiv,
        "reference_path": str(reference_path),
        "reference_keff": reference.keff,
        "published_keff": acceptance.published_keff,
        "keff": keff,
        "keff_error": keff_error,
        "flux_error": flux_error,
        "target_keff_error": acceptance.target_keff_error,
        "target_flux_error": acceptance.target_flux_error,
        "source_budget_checkpoint_complete": source_budget_complete,
        "source_budget_is_convergence_claim": False,
        "reference_free_converged": reference_free_converged,
        "accuracy_accepted": accuracy_accepted,
        "source_accuracy_accepted": source_accuracy_accepted,
        "extended_convergence_gate_passed": extended_convergence_gate_passed,
        "formal_comparison_accepted": extended_convergence_gate_passed,
        "acceptance_device": "cpu",
        "acceptance_seconds": time.perf_counter() - started,
    }
    _write_json_exclusive(acceptance_path, record)
    return record


def evaluate_locked_rfm_artifact(
    artifact_path: str | Path,
    acceptance_path: str | Path,
    acceptance: OfflineAcceptanceConfig,
    refinement_manifest_path: str | Path,
    *,
    algebraic_residual_tolerance: float = 1.0e-7,
    resume: bool = False,
) -> dict[str, object]:
    """Evaluate one immutable RFM draw after the common lock barrier."""
    artifact = Path(artifact_path)
    output = Path(acceptance_path)
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        return json.loads(output.read_text(encoding="utf-8"))
    artifact_sha256 = _sha256(artifact)
    with np.load(artifact, allow_pickle=False) as payload:
        if bool(payload["reference_used_for_production"]):
            raise RuntimeError("RFM production artifact used a reference")
        points = np.asarray(payload["field_points"], dtype=float)
        field = np.asarray(payload["normalized_field"], dtype=float)
        keff = float(payload["keff"])
        residual = float(payload["generalized_eigen_residual"])
        seed = int(payload["seed"])
    started = time.perf_counter()
    refinement, reference_path, reference = _validated_reference(
        acceptance, refinement_manifest_path
    )
    reference_flux = reference.flux_at(points)
    keff_error = relative_keff_error(keff, acceptance.published_keff)
    flux_error = relative_flux_max_error(field, reference_flux)
    reference_free_converged = bool(residual <= algebraic_residual_tolerance)
    accuracy_accepted = bool(
        keff_error <= acceptance.target_keff_error
        and flux_error <= acceptance.target_flux_error
    )
    record = {
        "method": "rfm",
        "seed": seed,
        "artifact": str(artifact),
        "artifact_sha256": artifact_sha256,
        "reference_access_stage": "offline_after_all_nn_and_rfm_locks",
        "reference_refinement_manifest": str(Path(refinement_manifest_path)),
        "reference_refinement_converged": refinement["refinement_converged"],
        "reference_subdiv": reference.subdiv,
        "reference_path": str(reference_path),
        "reference_keff": reference.keff,
        "published_keff": acceptance.published_keff,
        "keff": keff,
        "keff_error": keff_error,
        "flux_error": flux_error,
        "target_keff_error": acceptance.target_keff_error,
        "target_flux_error": acceptance.target_flux_error,
        "generalized_eigen_residual": residual,
        "algebraic_residual_tolerance": algebraic_residual_tolerance,
        "reference_free_converged": reference_free_converged,
        "accuracy_target_met_observed": accuracy_accepted,
        "accuracy_target_is_validity_gate": False,
        "common_post_lock_evaluation_complete": True,
        "audit_valid": reference_free_converged,
        "formal_comparison_accepted": reference_free_converged,
        "acceptance_device": "cpu",
        "acceptance_seconds": time.perf_counter() - started,
    }
    _write_json_exclusive(output, record)
    return record
