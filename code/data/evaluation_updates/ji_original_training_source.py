from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import eigh_tridiagonal

from rfm_paper_repro.paths import DATA


def potential_np(kind: str, x: np.ndarray) -> np.ndarray:
    if kind == "square":
        return x * x
    if kind == "exp":
        return np.exp(-math.pi * x)
    raise ValueError(kind)


def potential_torch(kind: str, x: torch.Tensor) -> torch.Tensor:
    if kind == "square":
        return x * x
    if kind == "exp":
        return torch.exp(-math.pi * x)
    raise ValueError(kind)


def reference_1d(kind: str, scale: float, n_grid: int = 6000, count: int = 4) -> np.ndarray:
    h = 1.0 / (n_grid + 1)
    x = h * np.arange(1, n_grid + 1)
    diag = 2.0 / h**2 + scale * potential_np(kind, x)
    off = -np.ones(n_grid - 1) / h**2
    return eigh_tridiagonal(diag, off, select="i", select_range=(0, count - 1))[0]


def cube_refs(kind: str, dim: int, scale: float, count: int = 3) -> list[float]:
    mu = reference_1d(kind, scale, count=max(4, count))
    refs = [dim * mu[0]]
    if count >= 2:
        refs.append((dim - 1) * mu[0] + mu[1])
    if count >= 3:
        refs.append((dim - 1) * mu[0] + mu[1])
    return refs[:count]


class ReLU2ResNet(nn.Module):
    """ReLU^2 ResNet used for the Ji et al. cube reproduction."""

    def __init__(self, dim: int, width: int, depth: int):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2")
        self.input = nn.Linear(dim, width)
        self.blocks = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 1))
        self.output = nn.Linear(width, 1)
        self.reset_parameters()

    @staticmethod
    def act(x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.input(x))
        for block in self.blocks:
            y = y + self.act(block(y))
        return self.output(y).squeeze(-1)


def sample_interior(batch: int, dim: int, device: torch.device) -> torch.Tensor:
    return torch.rand(batch, dim, device=device)


def sample_boundary(batch: int, dim: int, device: torch.device) -> torch.Tensor:
    x = torch.rand(batch, dim, device=device)
    faces = torch.randint(0, 2 * dim, (batch,), device=device)
    axes = torch.div(faces, 2, rounding_mode="floor")
    sides = (faces % 2).to(x.dtype)
    x[torch.arange(batch, device=device), axes] = sides
    return x


def sobol_interior(n: int, dim: int, device: torch.device, scramble_seed: int) -> torch.Tensor:
    engine = torch.quasirandom.SobolEngine(dim, scramble=True, seed=scramble_seed)
    return engine.draw(n).to(device=device)


def sobol_boundary(n: int, dim: int, device: torch.device, scramble_seed: int) -> torch.Tensor:
    per_face = int(math.ceil(n / (2 * dim)))
    engine = torch.quasirandom.SobolEngine(dim - 1, scramble=True, seed=scramble_seed)
    base = engine.draw(per_face * 2 * dim).to(device=device)
    out = torch.empty(per_face * 2 * dim, dim, device=device)
    row = 0
    for axis in range(dim):
        for side in (0.0, 1.0):
            chunk = base[row : row + per_face]
            coords = torch.empty(per_face, dim, device=device)
            coords[:, axis] = side
            other = [j for j in range(dim) if j != axis]
            coords[:, other] = chunk
            out[row : row + per_face] = coords
            row += per_face
    return out[:n]


def normalize_initial_output(net: nn.Module, dim: int, device: torch.device, target_mass: float, seed: int) -> None:
    if target_mass <= 0:
        return
    x = sobol_interior(4096, dim, device, seed)
    with torch.no_grad():
        mass = net(x).square().mean().item()
        if mass <= 1.0e-30:
            return
        factor = math.sqrt(target_mass / mass)
        if isinstance(net, ReLU2ResNet):
            net.output.weight.mul_(factor)
            net.output.bias.mul_(factor)


def clone_state_dict(net: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}


def is_admissible_candidate(
    stats: dict[str, float],
    *,
    mode: int,
    mass_floor: float,
    mass_ceiling: float,
    orth_ceiling: float,
) -> bool:
    required = ("loss", "mass", "orth_penalty", "rayleigh_eps")
    if not all(math.isfinite(float(stats[key])) for key in required):
        return False
    if not mass_floor <= stats["mass"] <= mass_ceiling:
        return False
    return mode == 1 or stats["orth_penalty"] <= orth_ceiling


def assert_reproduced_stats(
    expected: dict[str, float],
    actual: dict[str, float],
    *,
    context: str,
    rtol: float = 2.0e-6,
    atol: float = 2.0e-8,
) -> None:
    for key in ("loss", "a_eps", "mass", "boundary_integral", "orth_penalty", "rayleigh_eps"):
        target = float(expected[key])
        value = float(actual[key])
        if not math.isclose(value, target, rel_tol=rtol, abs_tol=atol):
            raise RuntimeError(
                f"{context}: checkpoint statistic {key} changed from {target:.12e} to {value:.12e}."
            )


def minimize_output_scale(net: nn.Module, stats: dict[str, float], eps2: float) -> float:
    """Exactly minimize the Ji loss along the scalar output-amplitude direction."""
    mass = float(stats["mass"])
    if mass <= 0.0:
        raise RuntimeError("Cannot scale a zero-mass network output.")
    effective_rayleigh = (float(stats["a_eps"]) + eps2 * float(stats["orth_penalty"])) / mass
    target_mass = 1.0 - effective_rayleigh / eps2
    if target_mass <= 0.0:
        raise RuntimeError("The selected state has no nonzero minimum along the output-amplitude direction.")
    factor = math.sqrt(target_mass / mass)
    with torch.no_grad():
        if not isinstance(net, ReLU2ResNet):
            raise TypeError("Output scaling is implemented for ReLU2ResNet only.")
        net.output.weight.mul_(factor)
        net.output.bias.mul_(factor)
    return factor


@dataclass
class Terms:
    loss: torch.Tensor
    a_eps: torch.Tensor
    domain_energy: torch.Tensor
    mass: torch.Tensor
    boundary_integral: torch.Tensor
    orth_penalty: torch.Tensor


def loss_terms(
    net: nn.Module,
    x: torch.Tensor,
    xb: torch.Tensor,
    *,
    kind: str,
    dim: int,
    scale: float,
    eps1: float,
    eps2: float,
    boundary_measure: float,
    previous: list[nn.Module],
    previous_norms: list[float],
) -> Terms:
    x = x.detach().requires_grad_(True)
    u = net(x)
    grad = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    v = scale * potential_torch(kind, x).sum(dim=1)
    domain_energy = (grad.square().sum(dim=1) + v * u.square()).mean()
    mass = u.square().mean()

    ub = net(xb)
    boundary_integral = boundary_measure * ub.square().mean()
    a_eps = domain_energy + eps1 * boundary_integral

    orth = torch.zeros((), device=x.device, dtype=x.dtype)
    for prev, norm in zip(previous, previous_norms):
        with torch.no_grad():
            up = prev(x)
        ip = (u * up).mean() / max(norm, 1.0e-14)
        orth = orth + ip.square()

    q = mass - 1.0
    loss = 0.5 * a_eps + 0.25 * eps2 * (q.square() + 2.0 * orth)
    return Terms(loss, a_eps, domain_energy, mass, boundary_integral, orth)


def estimate_terms(
    net: nn.Module,
    *,
    kind: str,
    dim: int,
    scale: float,
    eps1: float,
    eps2: float,
    boundary_measure: float,
    previous: list[nn.Module],
    previous_norms: list[float],
    interior: torch.Tensor,
    boundary: torch.Tensor,
    chunk: int,
) -> dict[str, float]:
    net.eval()
    for prev in previous:
        prev.eval()
    domain_energy = 0.0
    mass = 0.0
    orth_ips = [0.0 for _ in previous]
    n_total = interior.shape[0]

    for start in range(0, n_total, chunk):
        x = interior[start : start + chunk].detach().requires_grad_(True)
        u = net(x)
        grad = torch.autograd.grad(u.sum(), x, create_graph=False)[0]
        v = scale * potential_torch(kind, x).sum(dim=1)
        weight = x.shape[0] / n_total
        domain_energy += weight * (grad.square().sum(dim=1) + v * u.square()).mean().item()
        mass += weight * u.square().mean().item()
        for idx, (prev, norm) in enumerate(zip(previous, previous_norms)):
            with torch.no_grad():
                up = prev(x)
            orth_ips[idx] += weight * ((u.detach() * up).mean().item() / max(norm, 1.0e-14))

    b_total = boundary.shape[0]
    boundary_mean = 0.0
    with torch.no_grad():
        for start in range(0, b_total, chunk):
            xb = boundary[start : start + chunk]
            ub = net(xb)
            boundary_mean += (xb.shape[0] / b_total) * ub.square().mean().item()
    boundary_integral = boundary_measure * boundary_mean
    a_eps = domain_energy + eps1 * boundary_integral
    orth = float(sum(ip * ip for ip in orth_ips))
    q = mass - 1.0
    loss = 0.5 * a_eps + 0.25 * eps2 * (q * q + 2.0 * orth)
    return {
        "loss": loss,
        "a_eps": a_eps,
        "domain_energy": domain_energy,
        "mass": mass,
        "boundary_integral": boundary_integral,
        "orth_penalty": orth,
        "rayleigh_eps": a_eps / max(mass, 1.0e-30),
        "rayleigh_domain": domain_energy / max(mass, 1.0e-30),
        "norm": math.sqrt(max(mass, 0.0)),
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def train_kind(args: argparse.Namespace, kind: str, device: torch.device) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    refs = cube_refs(kind, args.dim, args.scale, count=args.modes)
    previous: list[nn.Module] = []
    previous_norms: list[float] = []
    progress: list[dict[str, object]] = []
    terminal: list[dict[str, object]] = []
    checkpoint_dir = DATA / f"{args.out_prefix}_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    live_progress_path = DATA / f"{args.out_prefix}_{kind}_progress_live.csv"
    if live_progress_path.exists():
        with live_progress_path.open(newline="", encoding="utf-8") as handle:
            progress = list(csv.DictReader(handle))

    eval_interior = sobol_interior(args.eval_interior, args.dim, device, args.seed + 1000 + 17 * len(kind))
    eval_boundary = sobol_boundary(args.eval_boundary, args.dim, device, args.seed + 2000 + 17 * len(kind))
    max_epochs = args.paper_epochs + args.tail_epochs

    for mode in range(1, args.modes + 1):
        torch.manual_seed(args.seed + 100 * mode + (0 if kind == "square" else 10000))
        net = ReLU2ResNet(args.dim, args.width, args.depth).to(device=device, dtype=torch.float32)
        completed_path = checkpoint_dir / f"{kind}_mode{mode}.pt"
        runtime_path = checkpoint_dir / f"{kind}_mode{mode}_runtime.pt"
        if completed_path.exists():
            completed = torch.load(completed_path, map_location=device, weights_only=False)
            final_row = completed["final_row"]
            if abs(float(final_row["lambda_ref"]) - refs[mode - 1]) > 1.0e-12 * abs(refs[mode - 1]):
                raise RuntimeError(f"Completed checkpoint {completed_path.name} uses a different reference value.")
            if int(final_row.get("accepted", 0)) != 1:
                raise RuntimeError(f"Completed checkpoint {completed_path.name} has not passed the acceptance audit.")
            net.load_state_dict(completed["state_dict"])
            reproduced = estimate_terms(
                net,
                kind=kind,
                dim=args.dim,
                scale=args.scale,
                eps1=args.eps1,
                eps2=args.eps2_square if kind == "square" else args.eps2_exp,
                boundary_measure=args.boundary_measure if args.boundary_measure > 0 else 2.0 * args.dim,
                previous=previous,
                previous_norms=previous_norms,
                interior=eval_interior,
                boundary=eval_boundary,
                chunk=args.eval_chunk,
            )
            assert_reproduced_stats(final_row, reproduced, context=completed_path.name)
            terminal.append(final_row)
            previous.append(net)
            previous_norms.append(float(final_row["norm"]))
            print(f"{kind} k={mode} reused completed converged checkpoint", flush=True)
            continue
        normalize_initial_output(net, args.dim, device, args.init_mass, args.seed + 3000 + 100 * mode)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            threshold=args.loss_rel_tol,
            threshold_mode="rel",
            min_lr=args.min_lr,
        )

        best_loss_eval: dict[str, float] | None = None
        best_loss_state: dict[str, torch.Tensor] | None = None
        best_error_eval: dict[str, float] | None = None
        best_error_state: dict[str, torch.Tensor] | None = None
        best_loss_epoch = 0
        last_improvement_epoch = 0
        recovery_count = 0
        stop_reason = "maximum_epochs"
        final_epoch = max_epochs
        start_epoch = 1
        elapsed_before = 0.0
        if runtime_path.exists():
            runtime = torch.load(runtime_path, map_location="cpu", weights_only=False)
            if abs(float(runtime["lambda_ref"]) - refs[mode - 1]) > 1.0e-12 * abs(refs[mode - 1]):
                raise RuntimeError(f"Runtime checkpoint {runtime_path.name} uses a different reference value.")
            net.load_state_dict(runtime["model_state"])
            opt.load_state_dict(runtime["optimizer_state"])
            scheduler.load_state_dict(runtime["scheduler_state"])
            best_loss_eval = runtime["best_loss_eval"]
            best_loss_state = runtime["best_loss_state"]
            best_error_eval = runtime["best_error_eval"]
            best_error_state = runtime["best_error_state"]
            best_loss_epoch = int(runtime["best_loss_epoch"])
            last_improvement_epoch = int(runtime["last_improvement_epoch"])
            recovery_count = int(runtime.get("recovery_count", 0))
            start_epoch = int(runtime["epoch"]) + 1
            elapsed_before = float(runtime["elapsed_seconds"])
            torch.set_rng_state(runtime["torch_rng_state"].cpu())
            if device.type == "cuda" and runtime.get("cuda_rng_state_all") is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in runtime["cuda_rng_state_all"]])
            progress = [
                row
                for row in progress
                if str(row["kind"]) != kind
                or int(row["mode"]) != mode
                or int(row["epoch"]) <= int(runtime["epoch"])
            ]
            print(f"{kind} k={mode} resumed at epoch={start_epoch}", flush=True)
        t0 = time.perf_counter()
        for epoch in range(start_epoch, max_epochs + 1):
            net.train()
            x = sample_interior(args.batch_interior, args.dim, device)
            xb = sample_boundary(args.batch_boundary, args.dim, device)
            eps1_epoch = args.eps1
            eps1_warmup = args.eps1_warmup_square if kind == "square" else args.eps1_warmup_exp
            if eps1_warmup > 0:
                eps1_epoch = args.eps1 * min(1.0, epoch / eps1_warmup)
            terms = loss_terms(
                net,
                x,
                xb,
                kind=kind,
                dim=args.dim,
                scale=args.scale,
                eps1=eps1_epoch,
                eps2=args.eps2_square if kind == "square" else args.eps2_exp,
                boundary_measure=args.boundary_measure if args.boundary_measure > 0 else 2.0 * args.dim,
                previous=previous,
                previous_norms=previous_norms,
            )
            opt.zero_grad(set_to_none=True)
            terms.loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            opt.step()

            if epoch % args.eval_every == 0 or epoch == 1 or epoch == max_epochs:
                eval_stats = estimate_terms(
                    net,
                    kind=kind,
                    dim=args.dim,
                    scale=args.scale,
                    eps1=args.eps1,
                    eps2=args.eps2_square if kind == "square" else args.eps2_exp,
                    boundary_measure=args.boundary_measure if args.boundary_measure > 0 else 2.0 * args.dim,
                    previous=previous,
                    previous_norms=previous_norms,
                    interior=eval_interior,
                    boundary=eval_boundary,
                    chunk=args.eval_chunk,
                )
                rel = abs(eval_stats["rayleigh_eps"] - refs[mode - 1]) / abs(refs[mode - 1])
                rel_domain = abs(eval_stats["rayleigh_domain"] - refs[mode - 1]) / abs(refs[mode - 1])
                eps2_now = args.eps2_square if kind == "square" else args.eps2_exp
                lambda_stab = eps2_now * (1.0 - eval_stats["mass"])
                stationarity_rel = abs(eval_stats["rayleigh_eps"] - lambda_stab) / max(
                    abs(eval_stats["rayleigh_eps"]), 1.0e-30
                )
                admissible = is_admissible_candidate(
                    eval_stats,
                    mode=mode,
                    mass_floor=args.mass_floor,
                    mass_ceiling=args.mass_ceiling,
                    orth_ceiling=args.candidate_orth_ceiling,
                )
                row = {
                    "kind": kind,
                    "mode": mode,
                    "epoch": epoch,
                    "seconds": elapsed_before + time.perf_counter() - t0,
                    "learning_rate": opt.param_groups[0]["lr"],
                    "paper_budget_reached": int(epoch >= args.paper_epochs),
                    "eps1_train": eps1_epoch,
                    "lambda_ref": refs[mode - 1],
                    "lambda_eps": eval_stats["rayleigh_eps"],
                    "lambda_domain": eval_stats["rayleigh_domain"],
                    "lambda_stabilized": lambda_stab,
                    "relerr_eps": rel,
                    "relerr_domain": rel_domain,
                    "relerr_stabilized": abs(lambda_stab - refs[mode - 1]) / abs(refs[mode - 1]),
                    "stationarity_rel": stationarity_rel,
                    "admissible": int(admissible),
                    **eval_stats,
                }
                progress.append(row)
                write_rows(DATA / f"{args.out_prefix}_{kind}_progress_live.csv", progress)
                if admissible and (
                    best_loss_eval is None or row["loss"] < best_loss_eval["loss"] * (1.0 - args.loss_rel_tol)
                ):
                    best_loss_eval = dict(row)
                    best_loss_state = clone_state_dict(net)
                    best_loss_epoch = epoch
                    last_improvement_epoch = epoch
                if admissible and (best_error_eval is None or row["relerr_eps"] < best_error_eval["relerr_eps"]):
                    best_error_eval = dict(row)
                    best_error_state = clone_state_dict(net)
                if epoch >= args.paper_epochs:
                    scheduler.step(float(row["loss"]))
                print(
                    f"{kind} k={mode} epoch={epoch:6d} "
                    f"lambda_eps={row['lambda_eps']:.8f} rel={rel:.3e} "
                    f"mass={row['mass']:.3e} bdy={row['boundary_integral']:.3e} "
                    f"lr={opt.param_groups[0]['lr']:.2e}",
                    flush=True,
                )
                if (
                    epoch >= args.collapse_after
                    and (not math.isfinite(eval_stats["mass"]) or eval_stats["mass"] < args.collapse_mass)
                    and best_loss_state
                ):
                    net.load_state_dict(best_loss_state)
                    opt.state.clear()
                    for group in opt.param_groups:
                        group["lr"] = max(args.min_lr, float(group["lr"]) * args.recovery_lr_factor)
                    recovery_count += 1
                    print(
                        f"{kind} k={mode} restored admissible checkpoint at epoch={best_loss_epoch} "
                        f"after mass collapse; recovery={recovery_count}",
                        flush=True,
                    )
                torch.save(
                    {
                        "lambda_ref": refs[mode - 1],
                        "epoch": epoch,
                        "elapsed_seconds": elapsed_before + time.perf_counter() - t0,
                        "model_state": net.state_dict(),
                        "optimizer_state": opt.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_loss_eval": best_loss_eval,
                        "best_loss_state": best_loss_state,
                        "best_error_eval": best_error_eval,
                        "best_error_state": best_error_state,
                        "best_loss_epoch": best_loss_epoch,
                        "last_improvement_epoch": last_improvement_epoch,
                        "recovery_count": recovery_count,
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    },
                    runtime_path,
                )
                if (
                    epoch >= args.paper_epochs
                    and opt.param_groups[0]["lr"] <= args.min_lr * 1.01
                    and epoch - last_improvement_epoch >= args.stop_patience
                ):
                    stop_reason = "validation_loss_plateau_at_minimum_learning_rate"
                    final_epoch = epoch
                    break

        selected_epoch = final_epoch
        selected_by = "final_epoch"
        selected_eval: dict[str, float] | None = None
        if args.selection == "best_error" and best_error_state is not None:
            net.load_state_dict(best_error_state)
            selected_epoch = int(best_error_eval["epoch"]) if best_error_eval is not None else final_epoch
            selected_by = "minimum_independent_eigenvalue_error"
            selected_eval = best_error_eval
        elif args.selection == "best_loss" and best_loss_state is not None:
            net.load_state_dict(best_loss_state)
            selected_epoch = int(best_loss_eval["epoch"]) if best_loss_eval is not None else final_epoch
            selected_by = "minimum_independent_validation_loss"
            selected_eval = best_loss_eval

        selected_stats = estimate_terms(
            net,
            kind=kind,
            dim=args.dim,
            scale=args.scale,
            eps1=args.eps1,
            eps2=args.eps2_square if kind == "square" else args.eps2_exp,
            boundary_measure=args.boundary_measure if args.boundary_measure > 0 else 2.0 * args.dim,
            previous=previous,
            previous_norms=previous_norms,
            interior=eval_interior,
            boundary=eval_boundary,
            chunk=args.eval_chunk,
        )
        if selected_eval is not None:
            assert_reproduced_stats(selected_eval, selected_stats, context=f"{kind} mode {mode} selected state")

        eps2_now = args.eps2_square if kind == "square" else args.eps2_exp
        output_scale_factor = minimize_output_scale(net, selected_stats, eps2_now)

        final_stats = estimate_terms(
            net,
            kind=kind,
            dim=args.dim,
            scale=args.scale,
            eps1=args.eps1,
            eps2=eps2_now,
            boundary_measure=args.boundary_measure if args.boundary_measure > 0 else 2.0 * args.dim,
            previous=previous,
            previous_norms=previous_norms,
            interior=eval_interior,
            boundary=eval_boundary,
            chunk=args.eval_chunk,
        )
        lambda_stabilized = eps2_now * (1.0 - final_stats["mass"])
        relerr_eps = abs(final_stats["rayleigh_eps"] - refs[mode - 1]) / abs(refs[mode - 1])
        stationarity_rel = abs(final_stats["rayleigh_eps"] - lambda_stabilized) / max(
            abs(final_stats["rayleigh_eps"]), 1.0e-30
        )
        accepted = (
            stop_reason == "validation_loss_plateau_at_minimum_learning_rate"
            and relerr_eps <= args.accept_relerr
            and args.mass_floor <= final_stats["mass"] <= args.mass_ceiling
            and (mode == 1 or final_stats["orth_penalty"] <= args.accept_orth)
            and stationarity_rel <= args.accept_stationarity
        )
        final_row = {
            "kind": kind,
            "mode": mode,
            "epoch": final_epoch,
            "selected_epoch": selected_epoch,
            "selected_by": selected_by,
            "stop_reason": stop_reason,
            "paper_epochs": args.paper_epochs,
            "tail_epochs_available": args.tail_epochs,
            "paper_interior_sample_size": args.train_interior,
            "paper_boundary_sample_size": args.train_boundary,
            "stochastic_batch_interior": args.batch_interior,
            "stochastic_batch_boundary": args.batch_boundary,
            "boundary_measure_multiplier": args.boundary_measure if args.boundary_measure > 0 else 2.0 * args.dim,
            "best_loss_epoch": best_loss_epoch,
            "recovery_count": recovery_count,
            "total_seconds": elapsed_before + time.perf_counter() - t0,
            "lambda_ref": refs[mode - 1],
            "lambda_eps": final_stats["rayleigh_eps"],
            "lambda_domain": final_stats["rayleigh_domain"],
            "lambda_stabilized": lambda_stabilized,
            "relerr_eps": relerr_eps,
            "relerr_domain": abs(final_stats["rayleigh_domain"] - refs[mode - 1]) / abs(refs[mode - 1]),
            "relerr_stabilized": abs(lambda_stabilized - refs[mode - 1]) / abs(refs[mode - 1]),
            "stationarity_rel": stationarity_rel,
            "output_scale_factor": output_scale_factor,
            "pre_scale_mass": selected_stats["mass"],
            "pre_scale_loss": selected_stats["loss"],
            "accepted": int(accepted),
            **final_stats,
        }
        terminal.append(final_row)
        write_rows(DATA / f"{args.out_prefix}_{kind}_terminal_live.csv", terminal)
        if not accepted:
            torch.save(
                {
                    "kind": kind,
                    "mode": mode,
                    "selected_epoch": selected_epoch,
                    "selected_by": selected_by,
                    "state_dict": clone_state_dict(net),
                    "final_row": final_row,
                },
                checkpoint_dir / f"{kind}_mode{mode}_failed.pt",
            )
            raise RuntimeError(
                f"{kind} mode {mode} failed acceptance: stop={stop_reason}, relerr={relerr_eps:.3e}, "
                f"orth={final_stats['orth_penalty']:.3e}, stationarity={stationarity_rel:.3e}."
            )
        torch.save(
            {
                "kind": kind,
                "mode": mode,
                "selected_epoch": selected_epoch,
                "selected_by": selected_by,
                "state_dict": clone_state_dict(net),
                "final_row": final_row,
            },
            checkpoint_dir / f"{kind}_mode{mode}.pt",
        )
        runtime_path.unlink(missing_ok=True)
        previous.append(net)
        previous_norms.append(final_stats["norm"])

    return progress, terminal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the Ji et al. 2024 Deep Ritz 10D cube experiment.")
    parser.add_argument("--kinds", nargs="+", default=["square", "exp"], choices=["square", "exp"])
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--scale", type=float, default=20.0, help="Potential multiplier; 20 matches Ji Fig. 7 reference levels.")
    parser.add_argument("--modes", type=int, default=3)
    parser.add_argument("--width", type=int, default=150)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--paper-epochs", type=int, default=100000)
    parser.add_argument("--tail-epochs", type=int, default=80000)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--eps1", type=float, default=2000.0)
    parser.add_argument("--eps1-warmup-square", type=int, default=0)
    parser.add_argument("--eps1-warmup-exp", type=int, default=2000)
    parser.add_argument("--eps2-square", type=float, default=1000.0)
    parser.add_argument("--eps2-exp", type=float, default=500.0)
    parser.add_argument("--batch-interior", type=int, default=8192)
    parser.add_argument("--batch-boundary", type=int, default=4096)
    parser.add_argument("--train-interior", type=int, default=200000)
    parser.add_argument("--train-boundary", type=int, default=80000)
    parser.add_argument("--eval-interior", type=int, default=200000)
    parser.add_argument("--eval-boundary", type=int, default=80000)
    parser.add_argument("--eval-chunk", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--init-mass", type=float, default=1.0)
    parser.add_argument(
        "--boundary-measure",
        type=float,
        default=1.0,
        help="Multiplier for the boundary sample mean; 1 reproduces the Ji et al. numerical convention.",
    )
    parser.add_argument("--seed", type=int, default=202406)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--selection", choices=["best_error", "best_loss", "final"], default="best_loss")
    parser.add_argument("--lr-factor", type=float, default=0.2)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument("--loss-rel-tol", type=float, default=1.0e-5)
    parser.add_argument("--stop-patience", type=int, default=20000)
    parser.add_argument("--mass-floor", type=float, default=0.10)
    parser.add_argument("--mass-ceiling", type=float, default=1.10)
    parser.add_argument("--collapse-mass", type=float, default=1.0e-3)
    parser.add_argument("--collapse-after", type=int, default=5000)
    parser.add_argument("--candidate-orth-ceiling", type=float, default=5.0e-2)
    parser.add_argument("--accept-orth", type=float, default=1.0e-2)
    parser.add_argument("--accept-stationarity", type=float, default=2.0e-2)
    parser.add_argument("--accept-relerr", type=float, default=3.0e-2)
    parser.add_argument("--recovery-lr-factor", type=float, default=0.2)
    parser.add_argument("--out-prefix", default="ji2024_drm_validated")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    all_progress: list[dict[str, object]] = []
    all_terminal: list[dict[str, object]] = []
    for kind in args.kinds:
        progress, terminal = train_kind(args, kind, device)
        all_progress.extend(progress)
        all_terminal.extend(terminal)
        write_rows(DATA / f"{args.out_prefix}_progress.csv", all_progress)
        write_rows(DATA / f"{args.out_prefix}_terminal.csv", all_terminal)
    metadata = {
        "source": "Ji et al., Journal of Scientific Computing 98 (2024), Article 48",
        "paper_configuration": {
            "epochs": args.paper_epochs,
            "learning_rate": args.lr,
            "width": args.width,
            "depth": args.depth,
            "interior_sample_size": args.train_interior,
            "boundary_sample_size": args.train_boundary,
            "eps1": args.eps1,
            "eps1_warmup_square": args.eps1_warmup_square,
            "eps1_warmup_exp": args.eps1_warmup_exp,
            "eps2_square": args.eps2_square,
            "eps2_exp": args.eps2_exp,
            "boundary_measure_multiplier": args.boundary_measure if args.boundary_measure > 0 else 2.0 * args.dim,
        },
        "implementation_batches": {
            "sampling": "fresh uniform Monte Carlo samples at every Adam step",
            "interior": args.batch_interior,
            "boundary": args.batch_boundary,
        },
        "convergence_tail": {
            "additional_epochs": args.tail_epochs,
            "maximum_total_epochs": args.paper_epochs + args.tail_epochs,
            "scheduler": "ReduceLROnPlateau",
            "minimum_learning_rate": args.min_lr,
            "stop_patience_epochs": args.stop_patience,
        },
        "selection": args.selection,
        "selection_rule": "minimum fixed-Sobol validation loss among admissible noncollapsed states",
        "output_amplitude_step": "exact one-dimensional minimization of the reported Ji loss",
        "acceptance": {
            "relative_eigenvalue_error": args.accept_relerr,
            "orthogonality_penalty": args.accept_orth,
            "stationarity_relative_residual": args.accept_stationarity,
            "mass_interval": [args.mass_floor, args.mass_ceiling],
            "requires_minimum_learning_rate_plateau": True,
        },
        "seed": args.seed,
        "torch": torch.__version__,
        "device": str(device),
    }
    (DATA / f"{args.out_prefix}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
