r"""Example 4 (Section 4.1): the IAEA quarter core.

The random feature space here is a plain cosine space on the whole square, with
no boundary factor: the reflective condition is satisfied by every cosine, and
the vacuum condition is natural for the weak form and enters as a boundary term.
Because the coefficients are constant on each cell, every entry of the pencil has
a closed form and the assembly is exact -- see :mod:`rfmeig.piecewise_constant`.
The frequencies are drawn uniformly from a triangle rather than isotropically,
which keeps a pair and its transpose from both appearing and makes the constant
mode the only one repeated.

Two numbers are reported: the relative error of the multiplication factor, and
the largest relative pointwise error of the flux after mean-power normalization.
The second is the harder one, and it is where the difference between methods
shows: the flux has a kink at every material interface, and how well a method
resolves it is exactly what that error measures.

``rfm``
    ``--seeds`` draws at the reported feature count;
``neural``
    trains one of the three baselines for the source paper's epoch budget;
``reference``
    solves the finite volume reference and stores it, which the other two need.

Run with::

    python -m rfmeig.experiments.exp4_iaea --method rfm
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigh

from rfmeig import piecewise_constant, provenance
from rfmeig.problems import iaea_reactor as reactor
from rfmeig.rayleigh_ritz import dominant_generalized_eigenpair

EXPERIMENT = "experiment4"

#: The reported feature count and frequency radius.
DEFAULT_FEATURES = 6000
DEFAULT_RADIUS = 105.0
#: The truncation threshold, relative to the largest energy eigenvalue.
DEFAULT_TRUNCATION = 1.0e-12
#: The twenty reporting draws.
DEFAULT_SEEDS = tuple(2026071200 + index for index in range(1, 21))
#: Refinement of the finite volume reference, in cells per material cell.
DEFAULT_REFINEMENT = 48

#: Epoch budgets of the three baselines, as their source paper sets them.
NEURAL_EPOCHS = {"drm": 61_750, "gipmnn": 419_000, "pc_gipmnn": 555_000}
NEURAL_LEARNING_RATE = 1.0e-3
NEURAL_WIDTH = 20


def sample_cosine_modes(count: int, seed: int, radius: float) -> np.ndarray:
    """A constant mode, then frequencies uniform on the triangle of the given side.

    Drawing from a triangle rather than a square is what avoids drawing both a
    frequency pair and its transpose: the two give the same cosine product up to
    a relabelling of the axes, and having both wastes a feature.  The constant
    mode is kept because it is the shape the flux is closest to.
    """
    if count < 2:
        raise ValueError("at least two features are needed")
    if radius <= 0.0:
        raise ValueError("the frequency radius must be positive")
    rng = np.random.default_rng(seed)
    unit = rng.random((count - 1, 2))
    folded = unit.sum(axis=1) > 1.0
    unit[folded] = 1.0 - unit[folded]
    return np.vstack((np.zeros((1, 2)), radius * unit))


def evaluate_field(
    points: np.ndarray, modes: np.ndarray, coefficients: np.ndarray, chunk: int = 4096
) -> np.ndarray:
    """The flux of a solved state, normalized to a peak of one and a positive sign."""
    alpha = np.pi * modes[:, 0] / reactor.DOMAIN_SIZE
    beta = np.pi * modes[:, 1] / reactor.DOMAIN_SIZE
    values = np.empty(points.shape[0], dtype=float)
    for start in range(0, points.shape[0], chunk):
        stop = min(points.shape[0], start + chunk)
        along_x = np.cos(points[start:stop, 0, None] * alpha[None, :])
        along_y = np.cos(points[start:stop, 1, None] * beta[None, :])
        values[start:stop] = (along_x * along_y) @ coefficients
    if float(values.mean()) < 0.0:
        values = -values
    return values / max(float(np.max(np.abs(values))), 1.0e-30)


def solve_rfm(
    features: int,
    seed: int,
    radius: float,
    truncation: float,
    *,
    direct_assembly: bool = False,
    matrix_free: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    r"""Draw, assemble, whiten and read off :math:`k_{\mathrm{eff}}`.

    The pencil is whitened by the loss matrix and the multiplication factor is
    the largest eigenvalue of the reduced fission matrix -- which is what
    :math:`k_{\mathrm{eff}}` is, read directly rather than as the reciprocal of a
    small eigenvalue.

    Three routes to the same numbers, in decreasing cost:

    ``direct_assembly``
        assembles cell by cell, which is the definition and the slowest;
    the default
        assembles by the rearrangement of :mod:`rfmeig.piecewise_constant` and
        forms the reduced matrix, which is what the recorded values were
        produced with;
    ``matrix_free``
        never forms the reduced matrix, taking the dominant pair from the
        operator instead;
    ``device``
        moves the assembly *and* the solve onto a card, which is what makes the
        timing comparable with the neural baselines -- those were trained on one.
        Leaving the assembly on the host would leave most of the work there and
        the device label would not be earned.

    The clock covers the whole method and stops before any error is measured.
    """
    stages: dict[str, float] = {}
    started = time.perf_counter()

    stage = time.perf_counter()
    modes = sample_cosine_modes(features, seed, radius)
    stages["draw_s"] = time.perf_counter() - stage

    alpha = np.pi * modes[:, 0] / reactor.DOMAIN_SIZE
    beta = np.pi * modes[:, 1] / reactor.DOMAIN_SIZE
    materials = reactor.cell_materials()
    diffusion, absorption, fission_map = reactor.material_fields(materials)
    edges = reactor.outer_edges()
    extra: dict[str, Any] = {}

    if device != "cpu":
        from rfmeig import gpu

        stage = time.perf_counter()
        loss, fission = gpu.piecewise_constant_pencil(
            alpha, beta, diffusion, absorption, fission_map, edges,
            reactor.CELL_SIZE, device=device,
        )
        stages["assemble_s"] = time.perf_counter() - stage

        stage = time.perf_counter()
        solved = gpu.solve_energy_whitened_dominant(
            loss, fission, tolerance=truncation, device=device
        )
        stages["eigensolve_s"] = time.perf_counter() - stage
        method_seconds = time.perf_counter() - started

        keff = solved["keff"]
        coefficients = solved["coefficients"]
        retained_count = solved["retained"]
        # The residual of the original pencil, measured on the host after the
        # device arrays are released, so that it is outside the method's clock.
        loss_host = loss.cpu().numpy()
        fission_host = fission.cpu().numpy()
        del loss, fission
        residual = _relative_residual(loss_host, fission_host, keff, coefficients)
        extra = {
            "power_iteration_residual": solved["power_iteration_residual"],
            "power_iterations": solved["power_iterations"],
            "device": device,
        }
    else:
        stage = time.perf_counter()
        assemble = (
            piecewise_constant.assemble_by_cells
            if direct_assembly
            else piecewise_constant.assemble
        )
        loss, fission = assemble(
            alpha, beta, diffusion, absorption, fission_map, edges, reactor.CELL_SIZE
        )
        stages["assemble_s"] = time.perf_counter() - stage

        stage = time.perf_counter()
        energy_values, energy_vectors = eigh(loss, check_finite=False)
        threshold = max(float(energy_values[-1]) * truncation, 1.0e-15)
        retained = energy_values > threshold
        whitening = (
            energy_vectors[:, retained] / np.sqrt(energy_values[retained])[None, :]
        )
        if matrix_free:
            keff, reduced_vector = dominant_generalized_eigenpair(fission, whitening)
            coefficients = whitening @ reduced_vector
        else:
            reduced = whitening.T @ fission @ whitening
            last = int(retained.sum()) - 1
            values, vectors = eigh(
                reduced, subset_by_index=[last, last], check_finite=False
            )
            coefficients = whitening @ vectors[:, 0]
            keff = float(values[0])
        stages["eigensolve_s"] = time.perf_counter() - stage
        method_seconds = time.perf_counter() - started

        retained_count = int(retained.sum())
        residual = _relative_residual(loss, fission, keff, coefficients)
        extra = {"device": "cpu"}

    return {
        "seed": seed,
        "features": features,
        "radius": radius,
        "retained": retained_count,
        "truncation": truncation,
        "keff": keff,
        "algebraic_residual": residual,
        "method_s": method_seconds,
        "modes": modes,
        "coefficients": coefficients,
        **stages,
        **extra,
    }


def _relative_residual(
    loss: np.ndarray, fission: np.ndarray, keff: float, coefficients: np.ndarray
) -> float:
    r""":math:`\|Fc-k Lc\|` against the size of its two terms.

    A criticality residual, so the fission operator plays the role the stiffness
    matrix plays elsewhere and the multiplication factor plays the role of the
    eigenvalue.
    """
    applied_fission = fission @ coefficients
    applied_loss = loss @ coefficients
    return float(
        np.linalg.norm(applied_fission - keff * applied_loss)
        / max(
            np.linalg.norm(applied_fission) + abs(keff) * np.linalg.norm(applied_loss),
            1.0e-30,
        )
    )


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------
def run_reference(args, run) -> list[dict[str, Any]]:
    started = time.perf_counter()
    reference = reactor.solve_reference(args.refinement)
    seconds = time.perf_counter() - started
    provenance.write_npz(
        run.path(f"reference_refinement{args.refinement}.npz"),
        keff=np.asarray(reference.keff),
        refinement=np.asarray(reference.refinement),
        labels=reference.labels,
        flux=reference.flux,
        layout_sha256=np.asarray(reactor.layout_sha256()),
    )
    print(f"reference k_eff = {reference.keff:.10f}  ({seconds:.1f}s)", flush=True)
    return [
        {
            "refinement": args.refinement,
            "keff": reference.keff,
            "cells": int((reference.labels > 0).sum()),
            "seconds": seconds,
        }
    ]


def _load_reference(path: Path) -> reactor.Reference:
    """The stored finite volume solution, under either spelling of its key.

    The archived references were written before ``subdiv`` was renamed to
    ``refinement``.  They are the solutions the reported errors are measured
    against, so they are read as they stand: regenerating one would move every
    error in the table, the baselines' included.
    """
    with np.load(path, allow_pickle=False) as stored:
        refinement = "refinement" if "refinement" in stored.files else "subdiv"
        return reactor.Reference(
            keff=float(stored["keff"]),
            refinement=int(stored[refinement]),
            labels=stored["labels"],
            flux=stored["flux"],
        )


def run_rfm(args, run) -> list[dict[str, Any]]:
    reference = _load_reference(args.reference)
    points = reactor.build_collocation().interior
    reference_flux = reference.flux_at(points)

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        call_id = f"rfm_seed{seed}"
        recorded = run.completed(call_id)
        if recorded is not None:
            rows.append(recorded)
            continue
        solved = solve_rfm(
            args.features,
            seed,
            args.radius,
            args.truncation,
            direct_assembly=args.direct_assembly,
            matrix_free=args.matrix_free,
            device=args.device,
        )
        modes = solved.pop("modes")
        coefficients = solved.pop("coefficients")
        started = time.perf_counter()
        field = evaluate_field(points, modes, coefficients)
        solved["field_evaluation_s"] = time.perf_counter() - started
        solved["keff_error"] = reactor.keff_error(solved["keff"], reference.keff)
        solved["flux_error"] = reactor.flux_error(field, reference_flux)
        provenance.write_npz(
            run.path("states", f"seed{seed}.npz"),
            modes=modes,
            coefficients=coefficients,
            field=field,
        )
        rows.append(run.complete(call_id, solved))
        print(
            f"seed {seed}: k_eff={solved['keff']:.9f} "
            f"error={solved['keff_error']:.3e} flux={solved['flux_error']:.3e} "
            f"{solved['method_s']:.1f}s",
            flush=True,
        )
    return rows


def run_neural(args, run) -> list[dict[str, Any]]:
    import torch

    from rfmeig.baselines import neural_reactor

    reference = _load_reference(args.reference)
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    data = reactor.build_collocation()
    tensors = neural_reactor.TensorCollocation.from_collocation(
        data, args.neural_method, device, dtype
    )
    reference_flux = reference.flux_at(data.interior)

    torch.manual_seed(args.neural_seed)
    np.random.seed(args.neural_seed)
    model = neural_reactor.ResidualFluxNetwork(
        width=NEURAL_WIDTH, outputs=neural_reactor.OUTPUTS[args.neural_method]
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    epochs = args.epochs or NEURAL_EPOCHS[args.neural_method]
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        parts = neural_reactor.objective(args.neural_method, model, tensors)
        parts["loss"].backward()
        optimizer.step()
        if epoch % args.log_every == 0 or epoch == epochs:
            record = {
                "epoch": epoch,
                **{key: float(value.detach()) for key, value in parts.items()},
                "seconds": time.perf_counter() - started,
            }
            history.append(record)
            print(
                f"  epoch {epoch:7d}  loss={record['loss']:.6e}  "
                f"{record['seconds']:.0f}s",
                flush=True,
            )
    training_seconds = time.perf_counter() - started

    keff = neural_reactor.evaluate_keff(model, tensors)
    field = neural_reactor.evaluate_flux(model, tensors.points, tensors.subdomains)
    row = {
        "method": args.neural_method,
        "seed": args.neural_seed,
        "epochs": epochs,
        "device": args.device,
        "dtype": args.dtype,
        "keff": keff,
        "keff_error": reactor.keff_error(keff, reference.keff),
        "flux_error": reactor.flux_error(field, reference_flux),
        "training_seconds": training_seconds,
    }
    provenance.write_csv(run.path(f"neural_{args.neural_method}_history.csv"), history)
    torch.save(
        {"state_dict": model.state_dict(), "row": row},
        run.path("states", f"{args.neural_method}_seed{args.neural_seed}.pt"),
    )
    print(
        f"{args.neural_method}: k_eff={keff:.9f} error={row['keff_error']:.3e} "
        f"flux={row['flux_error']:.3e} in {training_seconds:.0f}s",
        flush=True,
    )
    return [run.complete(f"neural_{args.neural_method}_{args.neural_seed}", row)]


def summarize_rfm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Median and range over the reporting draws, as the table quotes them."""
    keff_errors = np.asarray([float(row["keff_error"]) for row in rows])
    flux_errors = np.asarray([float(row["flux_error"]) for row in rows])
    times = np.asarray([float(row["method_s"]) for row in rows])
    return [
        {
            "method": "rfm",
            "resolution": int(rows[0]["features"]),
            "draws": len(rows),
            "keff": float(np.median([float(row["keff"]) for row in rows])),
            "keff_error": float(np.median(keff_errors)),
            "keff_error_min": float(keff_errors.min()),
            "keff_error_max": float(keff_errors.max()),
            "flux_error": float(np.median(flux_errors)),
            "flux_error_min": float(flux_errors.min()),
            "flux_error_max": float(flux_errors.max()),
            "seconds": float(np.median(times)),
        }
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--method", choices=("rfm", "neural", "reference"), default="rfm"
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threads", type=int, default=1)

    parser.add_argument("--features", type=int, default=DEFAULT_FEATURES)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--truncation", type=float, default=DEFAULT_TRUNCATION)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--direct-assembly",
        action="store_true",
        help="assemble by looping over cells; the same matrices, far more memory",
    )
    parser.add_argument(
        "--matrix-free",
        action="store_true",
        help="take the dominant mode from the operator instead of forming the "
        "reduced matrix; the same value, faster",
    )

    parser.add_argument("--refinement", type=int, default=DEFAULT_REFINEMENT)
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="the stored finite volume reference; required by --method rfm and neural",
    )

    parser.add_argument(
        "--neural-method", choices=("drm", "gipmnn", "pc_gipmnn"), default="drm"
    )
    parser.add_argument("--neural-seed", type=int, default=2026071104)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=NEURAL_LEARNING_RATE)
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu, or cuda to run the assembly and the solve on a card; the "
        "neural baselines were trained on one, so a device run is what makes "
        "the two timings comparable",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--log-every", type=int, default=1000)
    args = parser.parse_args(argv)

    provenance.require_threads(args.threads)
    if args.method != "reference" and args.reference is None:
        parser.error("--reference is required unless --method reference")

    configuration = {
        "method": args.method,
        "layout_sha256": reactor.layout_sha256(),
        "features": args.features,
        "frequency_radius": args.radius,
        "truncation": args.truncation,
        "seeds": list(args.seeds),
        "assembly": "closed-form cell integrals, no quadrature",
        "reference_refinement": args.refinement,
        "neural_method": args.neural_method,
        "neural_epochs": args.epochs or NEURAL_EPOCHS[args.neural_method],
        "device": args.device,
        "dtype": args.dtype,
    }
    run = provenance.open_run(
        EXPERIMENT, config=configuration, run_id=args.run_id, resume=args.resume
    )

    runner = {"rfm": run_rfm, "neural": run_neural, "reference": run_reference}[
        args.method
    ]
    rows = runner(args, run)
    provenance.write_csv(run.path(f"{args.method}_rows.csv"), rows)
    if args.method == "rfm":
        provenance.write_csv(run.path("rfm_summary.csv"), summarize_rfm(rows))

    run.seal(rows=len(rows))
    print(f"written to {run.directory}")


if __name__ == "__main__":
    main()
