from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from paper_assets.paths import EXPERIMENT4_RESULTS

from .config import RunConfig
from .geometry import CollocationData, build_collocation, geometry_sha256
from .networks import ResidualFluxNet
from .operators import (
    boundary_loss,
    interface_loss,
    inverse_iteration_state,
    strong_operator,
    weak_rayleigh,
)

RESULTS = EXPERIMENT4_RESULTS

TRAIN_FIELDS = (
    "run_id",
    "training_track",
    "method",
    "seed",
    "epoch",
    "learning_rate",
    "loss",
    "pde_loss",
    "boundary_loss",
    "interface_loss",
    "interior_points",
    "symmetry_points",
    "robin_points",
    "interface_points",
    "epoch_seconds",
    "cumulative_seconds",
    "device",
    "dtype",
)

VALIDATION_FIELDS = (
    "run_id",
    "training_track",
    "method",
    "seed",
    "epoch",
    "validation_metric",
    "equation_residual",
    "boundary_residual",
    "interface_residual",
    "validation_points",
    "validation_seconds",
    "cumulative_seconds",
    "learning_rate_after_schedule",
)


@dataclass
class _TensorCollocation:
    points: torch.Tensor
    labels: torch.Tensor
    subdomains: torch.Tensor | None
    symmetry_points: torch.Tensor
    symmetry_normals: torch.Tensor
    symmetry_labels: torch.Tensor
    symmetry_subdomains: torch.Tensor | None
    robin_points: torch.Tensor
    robin_normals: torch.Tensor
    robin_labels: torch.Tensor
    robin_subdomains: torch.Tensor | None
    interface_points: torch.Tensor
    interface_normals: torch.Tensor
    interface_left: torch.Tensor
    interface_right: torch.Tensor
    interface_left_material: torch.Tensor
    interface_right_material: torch.Tensor


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    # The pre-registered Example 4 seeds are date-stamped integers above
    # 2**32, which numpy's legacy seeder rejects.  Reduce only for numpy so the
    # recorded seed identifier stays the pre-registered one.
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _tensor(
    array: np.ndarray,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    value = torch.from_numpy(np.asarray(array))
    return value.to(device=device, dtype=dtype) if dtype is not None else value.to(device=device)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _append_csv_row(
    path: Path,
    fields: tuple[str, ...],
    row: dict[str, object],
    *,
    durable: bool = False,
) -> None:
    """Append one immutable record and reject schema drift.

    The training and validation streams are intentionally separate.  A resumed
    process never rewrites either stream, which preserves the original timing
    and optimization trajectory for audit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        if header != list(fields):
            raise RuntimeError(f"Refusing to append a changed schema to {path}")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        if durable:
            os.fsync(handle.fileno())


def _last_logged_epoch(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return max((int(row["epoch"]) for row in rows), default=0)


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


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


def _resume_log_paths(
    run_dir: Path,
    runtime: dict[str, object] | None,
) -> tuple[Path, Path, int]:
    if runtime is None:
        return run_dir / "train_epochs.csv", run_dir / "validation.csv", 0
    checkpoint_epoch = int(runtime["epoch"])
    old_train = run_dir / str(runtime.get("train_log_name", "train_epochs.csv"))
    old_validation = run_dir / str(runtime.get("validation_log_name", "validation.csv"))
    old_train_last = _last_logged_epoch(old_train)
    old_validation_last = _last_logged_epoch(old_validation)
    if old_train_last < checkpoint_epoch:
        raise RuntimeError("durable checkpoint is ahead of append-only training history")
    segment = int(runtime.get("history_segment", 0)) + 1
    _write_json_exclusive(
        run_dir / f"resume_event_{segment:03d}.json",
        {
            "checkpoint_epoch": checkpoint_epoch,
            "previous_train_log": old_train.name,
            "previous_train_last_epoch": old_train_last,
            "previous_validation_log": old_validation.name,
            "previous_validation_last_epoch": old_validation_last,
            "orphaned_train_tail_preserved": [checkpoint_epoch + 1, old_train_last],
            "recovery_policy": (
                "restart from the last durable runtime checkpoint; preserve, but do not "
                "treat later pre-crash rows as canonical"
            ),
            "new_history_segment": segment,
        },
    )
    return (
        run_dir / f"train_epochs.resume_{segment:03d}.csv",
        run_dir / f"validation.resume_{segment:03d}.csv",
        segment,
    )


def _tensor_collocation(
    data: CollocationData,
    device: torch.device,
    dtype: torch.dtype,
    *,
    global_method: bool,
) -> _TensorCollocation:
    subdomains = _tensor(data.subdomain, device).long()
    symmetry_subdomains = _tensor(data.symmetry_subdomain, device).long()
    robin_subdomains = _tensor(data.robin_subdomain, device).long()
    interface_left = _tensor(data.interface_left, device).long()
    interface_right = _tensor(data.interface_right, device).long()
    subdomain_material = torch.tensor(
        [3, 3, 3, 3, 2, 1, 4], device=device, dtype=torch.long
    )
    return _TensorCollocation(
        points=_tensor(data.interior, device, dtype),
        labels=_tensor(data.material, device).long(),
        subdomains=None if global_method else subdomains,
        symmetry_points=_tensor(data.symmetry_points, device, dtype),
        symmetry_normals=_tensor(data.symmetry_normals, device, dtype),
        symmetry_labels=_tensor(data.symmetry_material, device).long(),
        symmetry_subdomains=None if global_method else symmetry_subdomains,
        robin_points=_tensor(data.robin_points, device, dtype),
        robin_normals=_tensor(data.robin_normals, device, dtype),
        robin_labels=_tensor(data.robin_material, device).long(),
        robin_subdomains=None if global_method else robin_subdomains,
        interface_points=_tensor(data.interface_points, device, dtype),
        interface_normals=_tensor(data.interface_normals, device, dtype),
        interface_left=interface_left,
        interface_right=interface_right,
        interface_left_material=subdomain_material[interface_left],
        interface_right_material=subdomain_material[interface_right],
    )


def _fixed_validation_data(
    config: RunConfig,
    device: torch.device,
    dtype: torch.dtype,
    *,
    global_method: bool,
) -> _TensorCollocation:
    raw = build_collocation(
        nx=config.validation_nx,
        points_per_cell=config.validation_points_per_cell,
    )
    count = min(config.validation_interior_points, raw.interior.shape[0])
    rng = np.random.default_rng(config.seed + 17041)
    chosen = np.sort(rng.choice(raw.interior.shape[0], size=count, replace=False))
    fixed = replace(
        raw,
        interior=raw.interior[chosen],
        material=raw.material[chosen],
        subdomain=raw.subdomain[chosen],
    )
    return _tensor_collocation(fixed, device, dtype, global_method=global_method)


def _objective(
    config: RunConfig,
    model: ResidualFluxNet,
    data: _TensorCollocation,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    zero = torch.zeros((), device=data.points.device, dtype=data.points.dtype)
    if config.method == "drm":
        loss = weak_rayleigh(
            model,
            data.points,
            data.labels,
            data.subdomains,
            data.robin_points,
            data.robin_subdomains,
        )
        return loss, loss, zero, zero

    flux, applied, nu_sigma_f = strong_operator(
        model, data.points, data.labels, data.subdomains
    )
    eigenvalue_previous, previous = inverse_iteration_state(flux, applied, nu_sigma_f)
    pde = data.points.shape[0] * (
        applied - eigenvalue_previous * nu_sigma_f * previous
    ).square().mean()
    boundary = data.symmetry_points.shape[0] * boundary_loss(
        model,
        data.symmetry_points,
        data.symmetry_normals,
        data.symmetry_labels,
        data.symmetry_subdomains,
        data.robin_points,
        data.robin_normals,
        data.robin_labels,
        data.robin_subdomains,
    )
    interface = zero
    if config.method == "pc_gipmnn":
        continuity, current = interface_loss(
            model,
            data.interface_points,
            data.interface_normals,
            data.interface_left,
            data.interface_right,
            data.interface_left_material,
            data.interface_right_material,
        )
        interface = data.interface_points.shape[0] * (continuity + current)
    return pde + boundary + interface, pde, boundary, interface


def _validation_residual(
    config: RunConfig,
    model: ResidualFluxNet,
    data: _TensorCollocation,
) -> dict[str, float]:
    """Evaluate a fixed, reference-free equation residual.

    No FV flux, published k-effective value, or target error enters this
    calculation.  It is therefore admissible for scheduling, stopping, and
    checkpoint selection.
    """
    model.eval()
    with torch.enable_grad():
        flux, applied, nu_sigma_f = strong_operator(
            model, data.points, data.labels, data.subdomains
        )
        eigenvalue, previous = inverse_iteration_state(flux, applied, nu_sigma_f)
        source = eigenvalue * nu_sigma_f * previous
        equation = (applied - source).square().mean() / torch.clamp_min(
            source.square().mean(), 1.0e-30
        )
        boundary = boundary_loss(
            model,
            data.symmetry_points,
            data.symmetry_normals,
            data.symmetry_labels,
            data.symmetry_subdomains,
            data.robin_points,
            data.robin_normals,
            data.robin_labels,
            data.robin_subdomains,
        )
        interface = torch.zeros_like(equation)
        if config.method == "pc_gipmnn":
            continuity, current = interface_loss(
                model,
                data.interface_points,
                data.interface_normals,
                data.interface_left,
                data.interface_right,
                data.interface_left_material,
                data.interface_right_material,
            )
            interface = continuity + current
        metric = equation + boundary + interface
    values = {
        "validation_metric": float(metric.detach().cpu()),
        "equation_residual": float(equation.detach().cpu()),
        "boundary_residual": float(boundary.detach().cpu()),
        "interface_residual": float(interface.detach().cpu()),
    }
    model.zero_grad(set_to_none=True)
    model.train()
    return values


def _network_flux(
    model: ResidualFluxNet,
    points: torch.Tensor,
    subdomains: torch.Tensor | None,
    chunk: int = 8192,
) -> np.ndarray:
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, points.shape[0], chunk):
            stop = min(points.shape[0], start + chunk)
            selected = subdomains[start:stop] if subdomains is not None else None
            values.append(model.flux(points[start:stop], selected).detach().cpu().numpy())
    return np.concatenate(values)


def _evaluate_rayleigh(
    model: ResidualFluxNet,
    data: _TensorCollocation,
    chunk: int = 4096,
) -> float:
    model.eval()
    numerator = 0.0
    denominator = 0.0
    total = data.points.shape[0]
    from .geometry import active_area, material_arrays, robin_length

    for start in range(0, total, chunk):
        stop = min(total, start + chunk)
        selected = data.subdomains[start:stop] if data.subdomains is not None else None
        coordinates = data.points[start:stop].detach().requires_grad_(True)
        flux = model.flux(coordinates, selected)
        gradient = torch.autograd.grad(flux.sum(), coordinates, create_graph=False)[0]
        diffusion_np, sigma_a_np, nu_sigma_f_np = material_arrays(
            data.labels[start:stop].detach().cpu().numpy()
        )
        diffusion = torch.as_tensor(diffusion_np, device=data.points.device, dtype=data.points.dtype)
        sigma_a = torch.as_tensor(sigma_a_np, device=data.points.device, dtype=data.points.dtype)
        nu_sigma_f = torch.as_tensor(
            nu_sigma_f_np, device=data.points.device, dtype=data.points.dtype
        )
        weight = (stop - start) / total
        numerator += weight * float(
            (diffusion * gradient.square().sum(dim=1) + sigma_a * flux.square())
            .mean()
            .detach()
            .cpu()
        )
        denominator += weight * float(
            (nu_sigma_f * flux.square()).mean().detach().cpu()
        )
    with torch.no_grad():
        robin_flux = model.flux(data.robin_points, data.robin_subdomains)
        boundary_mean = float(robin_flux.square().mean().detach().cpu())
    eigenvalue = (
        active_area() * numerator + robin_length() * 0.5 * boundary_mean
    ) / (active_area() * denominator)
    return 1.0 / eigenvalue


def train_baseline(
    config: RunConfig,
    epoch_override: int | None = None,
    *,
    output_directory: str | Path | None = None,
    resume: bool = False,
) -> dict[str, object]:
    if epoch_override is not None:
        config = replace(
            config,
            paper_epochs=epoch_override,
            max_epochs=epoch_override,
            result_name=f"{config.result_name}_pilot{epoch_override}",
            training_track="source_faithful",
            scheduler_enabled=False,
            allow_early_stopping=False,
        )
    config.validate()
    set_determinism(config.seed)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no silent CPU fallback is allowed.")
    device = torch.device(config.device)
    dtype = _dtype(config.dtype)
    _synchronize(device)
    overall_started = time.perf_counter()

    run_dir = (
        Path(output_directory)
        if output_directory is not None
        else RESULTS / "fair_retrains" / config.result_name
    )
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"Refusing to overwrite existing run {run_dir}; choose a new run_id or pass resume=True."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name
    runtime_path = run_dir / "runtime.pt"
    locked_path = run_dir / "locked_model.pt"
    locked_field_path = run_dir / "locked_field.npz"
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    if resume:
        if not config_path.exists():
            raise RuntimeError(f"Cannot resume without immutable config in {run_dir}")
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config != asdict(config):
            raise RuntimeError("Resume configuration differs from the immutable recorded configuration")
        if summary_path.exists():
            completed = json.loads(summary_path.read_text(encoding="utf-8"))
            if _sha256(locked_path) != completed["locked_checkpoint_sha256"]:
                raise RuntimeError("completed locked checkpoint changed before resume")
            if _sha256(locked_field_path) != completed["locked_field_sha256"]:
                raise RuntimeError("completed locked field changed before resume")
            return completed
        if not runtime_path.exists():
            raise RuntimeError(f"Cannot resume incomplete run state in {run_dir}")
    else:
        with config_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(asdict(config), handle, indent=2)

    raw_train = build_collocation(
        nx=config.collocation_nx,
        points_per_cell=config.points_per_cell,
    )
    if config.collocation_nx == 171 and raw_train.interior.shape[0] != 24_240:
        raise RuntimeError(
            "IAEA geometry drift: nominal 171x171 grid must contain 24240 active points"
        )
    global_method = config.outputs == 1
    train_data = _tensor_collocation(raw_train, device, dtype, global_method=global_method)
    validation_data = _fixed_validation_data(
        config, device, dtype, global_method=global_method
    )
    data_manifest = {
        "sampling_mode": config.sampling_mode,
        "collocation_nx": config.collocation_nx,
        "nominal_grid_shape": [config.collocation_nx, config.collocation_nx],
        "nominal_tensor_grid_points": int(config.collocation_nx**2),
        "active_interior_points_used_each_epoch": int(train_data.points.shape[0]),
        "inactive_or_outside_points_excluded": int(
            config.collocation_nx**2 - train_data.points.shape[0]
        ),
        "nominal_171_squared_is_not_active_count": True,
        "symmetry_points_used_each_epoch": int(train_data.symmetry_points.shape[0]),
        "robin_points_used_each_epoch": int(train_data.robin_points.shape[0]),
        "interface_points_used_each_epoch": int(train_data.interface_points.shape[0]),
        "uses_all_fixed_training_points_each_epoch": True,
        "validation_nx": config.validation_nx,
        "fixed_validation_points": int(validation_data.points.shape[0]),
        "validation_reference_free": True,
        "geometry_sha256": geometry_sha256(),
    }
    manifest_path = run_dir / "data_manifest.json"
    if not resume:
        with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(data_manifest, handle, indent=2)
    elif json.loads(manifest_path.read_text(encoding="utf-8")) != data_manifest:
        raise RuntimeError("resume data manifest differs from regenerated geometry")

    model = ResidualFluxNet(config.width, config.outputs).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.2,
            patience=config.plateau_patience_evals,
            threshold=1.0e-6,
            threshold_mode="rel",
            min_lr=config.minimum_learning_rate,
        )
        if config.scheduler_enabled
        else None
    )
    _synchronize(device)

    start_epoch = 1
    elapsed_before = 0.0
    total_before = 0.0
    best_validation = float("inf")
    best_validation_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    runtime: dict[str, object] | None = None
    if resume:
        runtime = torch.load(runtime_path, map_location="cpu", weights_only=False)
        model.load_state_dict(runtime["model"])
        optimizer.load_state_dict(runtime["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(runtime["scheduler"])
        random.setstate(runtime["python_rng_state"])
        np.random.set_state(runtime["numpy_rng_state"])
        torch.set_rng_state(runtime["torch_rng_state"])
        if torch.cuda.is_available() and runtime.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(runtime["cuda_rng_state"])
        start_epoch = int(runtime["epoch"]) + 1
        elapsed_before = float(runtime["elapsed_seconds"])
        total_before = float(runtime.get("total_elapsed_seconds", elapsed_before))
        best_validation = float(runtime["best_validation"])
        best_validation_epoch = int(runtime["best_validation_epoch"])
        best_state = runtime["best_state"]
        print(f"resuming {config.method} at epoch {start_epoch}", flush=True)
    train_log_path, validation_log_path, history_segment = _resume_log_paths(
        run_dir, runtime
    )

    _synchronize(device)
    setup_seconds_current_process = time.perf_counter() - overall_started
    started = time.perf_counter()
    stop_reason = (
        "fixed_source_budget"
        if config.training_track == "source_faithful"
        else "maximum_extended_epochs"
    )
    final_epoch = start_epoch - 1
    last_validation: dict[str, float] | None = None

    for epoch in range(start_epoch, config.max_epochs + 1):
        model.train()
        _synchronize(device)
        epoch_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss, pde, boundary, interface = _objective(config, model, train_data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        _synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_started
        cumulative = total_before + time.perf_counter() - overall_started
        checkpoint_due = bool(
            epoch == 1
            or epoch % config.checkpoint_every == 0
            or epoch == config.max_epochs
        )
        train_row: dict[str, object] = {
            "run_id": run_id,
            "training_track": config.training_track,
            "method": config.method,
            "seed": config.seed,
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "loss": float(loss.detach().cpu()),
            "pde_loss": float(pde.detach().cpu()),
            "boundary_loss": float(boundary.detach().cpu()),
            "interface_loss": float(interface.detach().cpu()),
            "interior_points": int(train_data.points.shape[0]),
            "symmetry_points": int(train_data.symmetry_points.shape[0]),
            "robin_points": int(train_data.robin_points.shape[0]),
            "interface_points": int(train_data.interface_points.shape[0]),
            "epoch_seconds": epoch_seconds,
            "cumulative_seconds": cumulative,
            "device": str(device),
            "dtype": config.dtype,
        }
        _append_csv_row(
            train_log_path,
            TRAIN_FIELDS,
            train_row,
            durable=checkpoint_due,
        )

        evaluate = epoch == 1 or epoch % config.eval_every == 0 or epoch == config.max_epochs
        if evaluate:
            _synchronize(device)
            validation_started = time.perf_counter()
            last_validation = _validation_residual(config, model, validation_data)
            _synchronize(device)
            validation_seconds = time.perf_counter() - validation_started
            metric = last_validation["validation_metric"]
            if metric < best_validation * (1.0 - 1.0e-8):
                best_validation = metric
                best_validation_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            if scheduler is not None and epoch >= config.paper_epochs:
                scheduler.step(metric)
            validation_row: dict[str, object] = {
                "run_id": run_id,
                "training_track": config.training_track,
                "method": config.method,
                "seed": config.seed,
                "epoch": epoch,
                **last_validation,
                "validation_points": int(validation_data.points.shape[0]),
                "validation_seconds": validation_seconds,
                "cumulative_seconds": total_before + time.perf_counter() - overall_started,
                "learning_rate_after_schedule": optimizer.param_groups[0]["lr"],
            }
            _append_csv_row(
                validation_log_path,
                VALIDATION_FIELDS,
                validation_row,
                durable=checkpoint_due,
            )
            print(
                f"{config.method} track={config.training_track} epoch={epoch} "
                f"validation_residual={metric:.3e} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )

        final_epoch = epoch
        should_stop = bool(
            config.allow_early_stopping
            and epoch >= config.paper_epochs
            and scheduler is not None
            and optimizer.param_groups[0]["lr"] <= config.minimum_learning_rate * 1.01
            and epoch - best_validation_epoch >= config.stop_patience_epochs
        )
        if should_stop:
            stop_reason = "reference_free_validation_plateau_at_minimum_learning_rate"
        if checkpoint_due or should_stop:
            runtime_payload = {
                "config": asdict(config),
                "epoch": epoch,
                "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                "total_elapsed_seconds": total_before + time.perf_counter() - overall_started,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": None if scheduler is None else scheduler.state_dict(),
                "best_validation": best_validation,
                "best_validation_epoch": best_validation_epoch,
                "best_state": best_state,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
                "history_segment": history_segment,
                "train_log_name": train_log_path.name,
                "validation_log_name": validation_log_path.name,
                "recovery_granularity_epochs": config.checkpoint_every,
            }
            _atomic_torch_save(runtime_payload, runtime_path)
            if epoch % config.checkpoint_every == 0 or epoch == 1:
                _atomic_torch_save(
                    runtime_payload,
                    run_dir / "checkpoints" / f"epoch_{epoch:07d}.pt",
                )
        if should_stop:
            break

    if final_epoch < 1 or best_state is None or last_validation is None:
        raise RuntimeError("Training produced no lockable state and validation record.")
    training_validation_checkpoint_seconds = (
        elapsed_before + time.perf_counter() - started
    )

    if config.training_track == "source_faithful":
        locked_epoch = final_epoch
        locked_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        selected_by = "fixed_source_paper_epoch_budget"
    else:
        locked_epoch = best_validation_epoch
        locked_state = best_state
        selected_by = "minimum_fixed_reference_free_validation_residual"
    if locked_state is None:
        raise RuntimeError("no reference-free checkpoint was available to lock")
    model.load_state_dict(locked_state)
    locked_validation = _validation_residual(config, model, validation_data)
    reference_free_converged = bool(
        config.training_track == "extended_to_convergence"
        and stop_reason == "reference_free_validation_plateau_at_minimum_learning_rate"
        and np.isfinite(locked_validation["validation_metric"])
    )
    _synchronize(device)
    lock_started = time.perf_counter()
    _atomic_torch_save(
        {
            "config": asdict(config),
            "locked_epoch": locked_epoch,
            "selected_by": selected_by,
            "selection_uses_reference": False,
            "validation_metric": locked_validation["validation_metric"],
            "validation_components": locked_validation,
            "state_dict": locked_state,
        },
        locked_path,
    )
    _synchronize(device)
    locking_seconds = time.perf_counter() - lock_started

    _synchronize(device)
    field_started = time.perf_counter()
    locked_flux = _network_flux(model, train_data.points, train_data.subdomains)
    if float(np.mean(locked_flux)) < 0.0:
        locked_flux = -locked_flux
    locked_flux = locked_flux / max(float(np.max(np.abs(locked_flux))), 1.0e-30)
    locked_keff = _evaluate_rayleigh(model, train_data)
    _synchronize(device)
    field_core_seconds = time.perf_counter() - field_started
    serialization_started = time.perf_counter()
    with locked_field_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            points=raw_train.interior,
            subdomains=raw_train.subdomain,
            normalized_field=locked_flux,
            keff=np.asarray(locked_keff),
            locked_epoch=np.asarray(locked_epoch),
            reference_used=False,
            core_device=np.asarray(str(device)),
        )
    field_serialization_seconds = time.perf_counter() - serialization_started
    locked_field_sha256 = _sha256(locked_field_path)

    summary: dict[str, Any] = {
        "run_id": run_id,
        "method": config.method,
        "seed": config.seed,
        "training_track": config.training_track,
        "source_paper_epoch_budget": config.paper_epochs,
        "source_budget_checkpoint_complete": bool(
            config.training_track == "source_faithful"
            and final_epoch == config.paper_epochs
            and locked_epoch == config.paper_epochs
        ),
        "source_budget_is_convergence_claim": False,
        "stop_epoch": final_epoch,
        "stop_reason": stop_reason,
        "stop_seconds": training_validation_checkpoint_seconds,
        "setup_seconds_current_process": setup_seconds_current_process,
        "training_validation_checkpoint_seconds": training_validation_checkpoint_seconds,
        "locking_seconds": locking_seconds,
        "selected_epoch": locked_epoch,
        "selected_by": selected_by,
        "selection_uses_reference": False,
        "scheduler_metric": (
            "fixed_reference_free_equation_residual"
            if config.scheduler_enabled
            else "disabled_fixed_source_budget"
        ),
        "stopping_metric": (
            "fixed_reference_free_equation_residual"
            if config.allow_early_stopping
            else "fixed_source_paper_epoch_budget"
        ),
        "best_validation_metric": best_validation,
        "locked_validation_metric": locked_validation["validation_metric"],
        "locked_validation_components": locked_validation,
        "reference_free_converged": reference_free_converged,
        "reference_free_convergence_required_for_formal_comparison": True,
        "geometry_sha256": geometry_sha256(),
        "device_core": str(device),
        "device_preprocessing": "cpu_then_explicit_transfer",
        "device_field_evaluation": str(device),
        "device_artifact_serialization": "cpu",
        "dtype": config.dtype,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "data_manifest": data_manifest,
        "evaluation_status": "locked_unevaluated",
        "reference_accessed_during_training": False,
        "locked_checkpoint": str(locked_path),
        "locked_checkpoint_sha256": _sha256(locked_path),
        "locked_field": str(locked_field_path),
        "locked_field_sha256": locked_field_sha256,
        "locked_field_core_seconds": field_core_seconds,
        "locked_field_serialization_seconds": field_serialization_seconds,
        "history_segment": history_segment,
        "recovery_granularity_epochs": config.checkpoint_every,
    }
    summary["total_wall_seconds"] = total_before + time.perf_counter() - overall_started
    if device.type == "cuda":
        summary["gpu"] = torch.cuda.get_device_name(device)
        summary["gpu_memory_bytes"] = torch.cuda.get_device_properties(device).total_memory
    with summary_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2)
    runtime_path.unlink(missing_ok=True)
    return summary
