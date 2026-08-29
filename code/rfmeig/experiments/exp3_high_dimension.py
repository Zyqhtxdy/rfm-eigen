r"""Example 3 (Section 4.1): the ten-dimensional benchmark of Ji et al.

Two methods, one integration rule.  In ten dimensions the rule is not a detail:
neither method can integrate exactly, both are estimating the same integrals
from two hundred thousand quasi-random points, and if they were given different
points the comparison would be partly a comparison of quadratures.  So the
random feature method assembles on the same scrambled Sobol nodes the neural
baseline validates on, and the neural baseline's trained states are re-evaluated
on those same nodes -- with the same scramble -- rather than quoted from their own
run.

``rfm``
    ``--trials`` draws at each feature count, on the shared rule;
``drm``
    the six stored states of the neural baseline, re-evaluated.  Nothing is
    retrained: the states are the ones the source-paper reproduction produced,
    and re-evaluating them changes only which rule the number comes from.

Two densities are available for the rule.  ``uniform`` is the plain Sobol rule.
``beta22`` sends each coordinate through the inverse Beta(2,2) distribution and
divides by its density, which is exact for the same integrals and puts more
points where the boundary factor carries its mass.  The reported table uses
``beta22`` for both methods; the uniform path is run as a check that the change
of density reproduces what the stored states were originally validated with, and
that check is what the ``uniform_reproduces_stored`` column records.

Run with::

    python -m rfmeig.experiments.exp3_high_dimension --method rfm
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from rfmeig import provenance, quadrature
from rfmeig.problems.high_dimensional_cube import (
    BOUNDARY_POINTS,
    INTEGRATION_POINTS,
    HighDimensionalCube,
)
from rfmeig.rayleigh_ritz import (
    assemble_cosine_pencil_expanded,
    solve_mass_whitened,
)

EXPERIMENT = "experiment3"

DEFAULT_POTENTIALS = ("square", "exp")
DEFAULT_FEATURE_COUNTS = (128, 256, 384, 512)
DEFAULT_TRIALS = 20
DEFAULT_BASE_SEED = 20260603
#: How many eigenvalues are reported.  The second and third are one double level.
DEFAULT_COUNT = 3
#: Larger than the threshold of the other experiments by many orders, because the
#: pencil here is assembled by quasi-Monte Carlo: the integration error, not the
#: arithmetic, is what limits which directions carry information.
DEFAULT_TRUNCATION = 1.0e-5
DEFAULT_BLOCK = 4096


def draw_seed(base: int, name: str, count: int, trial: int, scale: float) -> int:
    """The seed of one draw; the potential enters through the length of its name.

    Not elegant, but it is the rule the recorded runs used, and changing it would
    change every realization.
    """
    return base + 100_000 * len(name) + 1_000 * count + 97 * trial + int(100 * scale)


def run_rfm_trial(
    problem: HighDimensionalCube,
    points: np.ndarray,
    weights: np.ndarray,
    feature_count: int,
    seed: int,
    *,
    count: int,
    truncation: float,
    block: int,
    device: str = "cpu",
) -> dict[str, Any]:
    """One realization on the shared rule, timed from the draw to the eigenvalues.

    ``device`` moves the assembly and the solve onto a card.  The neural
    baseline here was trained on one, so a device run is what makes a timing
    comparison between the two meaningful -- though Table 3 reports errors only.
    """
    reaction = problem.reaction(points)

    started = time.perf_counter()
    omega, phase = problem.draw_features(feature_count, seed)
    drawn = time.perf_counter()
    if device != "cpu":
        from rfmeig import gpu

        energy, mass = gpu.cosine_pencil_expanded(
            points, weights, omega, phase, problem.boundary_factor,
            reaction=reaction, block=block, device=device,
        )
        assembled = time.perf_counter()
        solved = gpu.solve_mass_whitened(
            energy, mass, count, tolerance=truncation, device=device
        )
        finished = time.perf_counter()
        eigenvalues = solved["eigenvalues"]
        retained = solved["retained"]
    else:
        energy, mass = assemble_cosine_pencil_expanded(
            points,
            weights,
            omega,
            phase,
            problem.boundary_factor,
            reaction=reaction,
            block=block,
        )
        assembled = time.perf_counter()
        solution = solve_mass_whitened(energy, mass, count, tolerance=truncation)
        finished = time.perf_counter()
        eigenvalues = solution.eigenvalues
        retained = solution.retained

    row: dict[str, Any] = {
        "potential": problem.name,
        "N": feature_count,
        "seed": seed,
        "retained": retained,
        "device": device,
        "largest_frequency": float(np.max(np.linalg.norm(omega, axis=1))),
        "draw_s": drawn - started,
        "assemble_s": assembled - drawn,
        "eigensolve_s": finished - assembled,
        "method_s": finished - started,
    }
    for index, value in enumerate(eigenvalues, start=1):
        row[f"lambda_{index}"] = float(value)
    return row


def run_rfm(args, run, rule_cache) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in args.potentials:
        problem = HighDimensionalCube(name)
        points, weights = rule_cache(problem, args.density)
        reference = problem.reference_levels(args.count, grid=args.reference_grid)
        printed = ", ".join(f"{value:.9f}" for value in reference)
        print(f"{name}: reference {printed}", flush=True)
        for feature_count in args.feature_counts:
            for trial in range(args.trials):
                call_id = f"rfm_{name}_N{feature_count:05d}_trial{trial:03d}"
                recorded = run.completed(call_id)
                if recorded is not None:
                    rows.append(recorded)
                    continue
                row = run_rfm_trial(
                    problem,
                    points,
                    weights,
                    feature_count,
                    draw_seed(
                        args.base_seed,
                        name,
                        feature_count,
                        trial,
                        problem.frequency_scale,
                    ),
                    count=args.count,
                    truncation=args.truncation,
                    block=args.block,
                    device=args.device,
                )
                row["trial"] = trial
                for index, exact in enumerate(reference, start=1):
                    row[f"lambda_ref_{index}"] = exact
                    row[f"rel_error_{index}"] = (
                        abs(row[f"lambda_{index}"] - exact) / exact
                    )
                rows.append(run.complete(call_id, row))
                errors = " ".join(
                    f"{row[f'rel_error_{index}']:.2e}"
                    for index in range(1, args.count + 1)
                )
                print(
                    f"{name} N={feature_count:4d} trial={trial} "
                    f"errors {errors}  {row['method_s']:.1f}s",
                    flush=True,
                )
    return rows


def run_drm(args, run, rule_cache) -> list[dict[str, Any]]:
    from rfmeig.baselines import deep_ritz_cube

    rows: list[dict[str, Any]] = []
    for name in args.potentials:
        problem = HighDimensionalCube(name)
        reference = problem.reference_levels(args.count, grid=args.reference_grid)
        uniform_points, uniform_weights = rule_cache(problem, "uniform")
        density_points, density_weights = rule_cache(problem, args.density)
        boundary = deep_ritz_cube.sobol_boundary(
            args.boundary_points, problem.dimension, problem.boundary_scramble_seed
        )

        for mode in range(1, args.count + 1):
            path = args.checkpoints / f"{name}_mode{mode}.pt"
            model, metadata = deep_ritz_cube.load_checkpoint(path, problem.dimension)
            started = time.perf_counter()
            uniform = deep_ritz_cube.evaluate_state(
                model,
                name,
                problem.scale,
                uniform_points,
                uniform_weights,
                boundary_points=boundary,
                chunk=args.block,
            )
            shifted = deep_ritz_cube.evaluate_state(
                model,
                name,
                problem.scale,
                density_points,
                density_weights,
                boundary_points=boundary,
                chunk=args.block,
            )
            exact = reference[mode - 1]
            row = {
                "potential": name,
                "mode": mode,
                "checkpoint": path.name,
                "lambda_ref": exact,
                "lambda_uniform": uniform.penalized_rayleigh,
                f"lambda_{args.density}": shifted.penalized_rayleigh,
                "rel_error_uniform": abs(uniform.penalized_rayleigh - exact) / exact,
                f"rel_error_{args.density}": abs(shifted.penalized_rayleigh - exact)
                / exact,
                # The interior quotient omits the boundary penalty.  It is not
                # the reported number -- the iterate is not zero on the boundary,
                # so it is not an approximate eigenvalue -- but the gap between
                # the two says how much of the state sits outside the space.
                "lambda_interior_uniform": uniform.interior_rayleigh,
                "boundary_integral": uniform.boundary_integral,
                "evaluation_seconds": time.perf_counter() - started,
                **metadata,
            }
            rows.append(run.complete(f"drm_{name}_mode{mode}", row))
            print(
                f"{name} mode {mode}: {args.density} "
                f"{row[f'lambda_{args.density}']:.6f} "
                f"(error {row[f'rel_error_{args.density}']:.3e})",
                flush=True,
            )
    return rows


def summarize_rfm(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Median and range of the relative error of each level, at each feature count."""
    summary: list[dict[str, Any]] = []
    keys = sorted({(row["potential"], int(row["N"])) for row in rows})
    for name, feature_count in keys:
        block = [
            row
            for row in rows
            if row["potential"] == name and int(row["N"]) == feature_count
        ]
        for index in range(1, count + 1):
            errors = np.asarray([float(row[f"rel_error_{index}"]) for row in block])
            values = np.asarray([float(row[f"lambda_{index}"]) for row in block])
            summary.append(
                {
                    "potential": name,
                    "N": feature_count,
                    "eigen_index": index,
                    "trials": len(block),
                    "lambda_ref": float(block[0][f"lambda_ref_{index}"]),
                    "lambda_median": float(np.median(values)),
                    "rel_error_median": float(np.median(errors)),
                    "rel_error_min": float(errors.min()),
                    "rel_error_max": float(errors.max()),
                    "retained_median": float(
                        np.median([row["retained"] for row in block])
                    ),
                    "method_s_median": float(
                        np.median([row["method_s"] for row in block])
                    ),
                }
            )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=("rfm", "drm"), default="rfm")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threads", type=int, default=1)

    parser.add_argument(
        "--potentials",
        nargs="+",
        choices=DEFAULT_POTENTIALS,
        default=list(DEFAULT_POTENTIALS),
    )
    parser.add_argument(
        "--feature-counts", type=int, nargs="+", default=list(DEFAULT_FEATURE_COUNTS)
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--truncation", type=float, default=DEFAULT_TRUNCATION)
    parser.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu, or cuda to assemble and solve on a card",
    )
    parser.add_argument("--integration-points", type=int, default=INTEGRATION_POINTS)
    parser.add_argument(
        "--density", choices=("beta22", "beta33", "uniform"), default="beta22"
    )
    parser.add_argument("--reference-grid", type=int, default=20_000)
    parser.add_argument("--boundary-points", type=int, default=BOUNDARY_POINTS)
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=Path("data/ji2024_drm_validated_checkpoints"),
    )
    args = parser.parse_args(argv)

    provenance.require_threads(args.threads)

    cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def rule_cache(problem, density):
        """The rule is shared by every trial of a potential, so build it once."""
        key = (problem.name, density)
        if key not in cache:
            cache[key] = quadrature.sobol_cube(
                args.integration_points,
                problem.dimension,
                problem.scramble_seed,
                density=density,
            )
        return cache[key]

    configuration = {
        "method": args.method,
        "dimension": 10,
        "potential_scale": 20.0,
        "potentials": list(args.potentials),
        "integration": (
            f"scrambled Sobol, {args.integration_points} points, {args.density} density"
        ),
        "scramble_seed_rule": "203406 + 17 * len(potential)",
        "feature_counts": list(args.feature_counts),
        "trials": args.trials,
        "seed_rule": f"{args.base_seed}+100000*len(potential)+1000*N+97*trial+200",
        "truncation_tolerance": args.truncation,
        "device": args.device,
        "reference_grid": args.reference_grid,
        "reference_rule": "second-order finite difference, tensor sums by heap",
        "eigenvalues_reported": args.count,
    }
    run = provenance.open_run(
        EXPERIMENT, config=configuration, run_id=args.run_id, resume=args.resume
    )

    if args.method == "rfm":
        rows = run_rfm(args, run, rule_cache)
        provenance.write_csv(run.path("rfm_trials.csv"), rows)
        provenance.write_csv(
            run.path("rfm_summary.csv"), summarize_rfm(rows, args.count)
        )
    else:
        rows = run_drm(args, run, rule_cache)
        provenance.write_csv(run.path("drm_reevaluation.csv"), rows)

    run.seal(rows=len(rows))
    print(f"written to {run.directory}")


if __name__ == "__main__":
    main()
