r"""Example 5 (Section 4.2): a shifted Gross--Pitaevskii ground state.

The problem is nonlinear, so nothing in Section 3 applies to it directly.  It is
here for a different reason: to show that a conforming random feature space is
usable as the trial space of a *constrained minimization* as well as of a linear
eigenvalue problem, and that being conforming buys the same thing in both cases.
The energy is minimized over the unit sphere of :math:`L^2` by the Riemannian
method of :mod:`rfmeig.nonlinear`, and the eigenvalue reported is the Lagrange
multiplier of the constraint.

The comparison is against the weak Galerkin method of Lu and Zhai, reimplemented
in :mod:`rfmeig.baselines.weak_galerkin`, at the four mesh resolutions their
table reports.  Both methods are run on the same machine on one thread, both are
timed over construction and solve together, and each configuration is timed
several times with the methods interleaved, so that a drift in the machine
affects both equally.  A warm-up call of each method is discarded before any
timing begins.

The reference is neither method's own value.  It is an independent spectral
solution, quoted here to ten digits, and both methods are measured against it.
The weak Galerkin values at the finest mesh are additionally compared against the
source's printed table, which is how the reimplementation is checked.

Run with::

    python -m rfmeig.experiments.exp5_gpe --run-id my-run
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from rfmeig import provenance
from rfmeig.baselines.weak_galerkin import WeakGalerkinSettings, solve_weak_galerkin
from rfmeig.box_basis import BoxFeatureBasis
from rfmeig.final_evaluation import scalar_gpe_observables
from rfmeig.nonlinear import RiemannianSettings, solve_scalar_riemannian
from rfmeig.problems.condensate import get_paper_benchmark

EXPERIMENT = "experiment5"
BENCHMARK = "E7_shifted_400"

#: The independent spectral reference both methods are measured against.
REFERENCE_LAMBDA = 16.6299951125
REFERENCE_ENERGY = 5.8505871135
#: The source's own printed values, at the finest mesh it reports.
SOURCE_TABLE = {"subdivisions": 128, "lambda": 16.629569, "energy": 5.850096}

DEFAULT_FEATURE_COUNTS = (128, 256, 384, 512, 768)
DEFAULT_SEEDS = (2086155103, 2086155137, 2086155179, 2086155221, 2086155263)
DEFAULT_SUBDIVISIONS = (16, 32, 64, 128)
DEFAULT_REPEATS = 5
#: Discarded, so that the first timed call is not the one that pays for the
#: library's first allocation of every buffer it will reuse.
WARMUP_SEED = 1086155001

#: The feature law, fixed before the runs and not tuned against the reference.
FEATURE_WIDTH = 0.09
CENTER_SPREAD = 0.3
QUADRATURE_ORDER = 56
EVALUATION_ORDER = 84
FINAL_EVALUATION_ORDER = 112
MASS_CUTOFF = 1.0e-12
MAX_ITERATIONS = 220
RESIDUAL_TOLERANCE = 1.0e-10

#: Set to zero: the classical stabilization; see the baseline's module docstring.
WEAK_GALERKIN_EPSILON = 0.0


def rfm_settings() -> RiemannianSettings:
    return RiemannianSettings(
        quadrature_order=QUADRATURE_ORDER,
        evaluation_order=EVALUATION_ORDER,
        mass_cutoff=MASS_CUTOFF,
        max_iterations=MAX_ITERATIONS,
        residual_tolerance=RESIDUAL_TOLERANCE,
        maximum_backtracks=30,
    )


def timed_call(construct, solve) -> tuple[Any, Any, float]:
    """Time construction and solve together, which is what a user pays.

    Timing the solve alone would flatter whichever method hides more of its work
    in setting up -- a mesh for one, a feature realization and its Gram matrices
    for the other -- so the clock covers both and stops before anything is
    written to disk.
    """
    started = time.perf_counter()
    built = construct()
    solved = solve(built)
    return built, solved, time.perf_counter() - started


def run_rfm_call(features: int, seed: int) -> tuple[Any, Any, float]:
    benchmark = get_paper_benchmark(BENCHMARK)
    settings = rfm_settings()
    return timed_call(
        lambda: BoxFeatureBasis.sample(
            "rbf",
            int(features),
            FEATURE_WIDTH,
            int(seed),
            benchmark.bounds,
            center_spread=CENTER_SPREAD,
        ),
        lambda basis: solve_scalar_riemannian(
            benchmark, basis, settings, capture_history=True
        ),
    )


def run_weak_galerkin_call(subdivisions: int, epsilon: float) -> tuple[Any, Any, float]:
    benchmark = get_paper_benchmark(BENCHMARK)
    return timed_call(
        lambda: WeakGalerkinSettings(
            subdivisions=int(subdivisions),
            epsilon=float(epsilon),
            max_iterations=600,
            gradient_tolerance=1e-8,
        ),
        lambda settings: solve_weak_galerkin(benchmark, settings, capture_history=True),
    )


def _rfm_row(features: int, seed: int, result, seconds: float, *, basis) -> dict[str, Any]:
    final = scalar_gpe_observables(
        get_paper_benchmark(BENCHMARK), basis, result.coefficients,
        order=FINAL_EVALUATION_ORDER,
    )
    return {
        "method": "RFM",
        "features": int(features),
        "seed": int(seed),
        "retained": int(result.retained_rank),
        **final,
        "optimization_lambda": float(result.eigenvalue),
        "optimization_energy": float(result.energy),
        "lambda_abs_error": abs(final["lambda"] - REFERENCE_LAMBDA),
        "energy_abs_error": abs(final["energy"] - REFERENCE_ENERGY),
        "weak_residual": float(result.weak_residual),
        "pde_residual_rms": float(result.pde_residual_rms),
        "iterations": int(result.iterations),
        "converged": bool(result.converged),
        "wall_seconds": seconds,
    }


def _weak_galerkin_row(
    subdivisions: int, epsilon: float, result, seconds: float
) -> dict[str, Any]:
    row = {
        "method": "weak Galerkin",
        "subdivisions": int(subdivisions),
        "epsilon": float(epsilon),
        "unknowns": int(result.degrees_of_freedom),
        "lambda": float(result.eigenvalue),
        "energy": float(result.energy),
        "lambda_abs_error": abs(float(result.eigenvalue) - REFERENCE_LAMBDA),
        "energy_abs_error": abs(float(result.energy) - REFERENCE_ENERGY),
        "weak_residual": float(result.residual_norm),
        "iterations": int(result.iterations),
        "converged": bool(result.converged),
        "wall_seconds": seconds,
    }
    if int(subdivisions) == SOURCE_TABLE["subdivisions"]:
        row["lambda_abs_error_to_source"] = abs(
            float(result.eigenvalue) - SOURCE_TABLE["lambda"]
        )
        row["energy_abs_error_to_source"] = abs(
            float(result.energy) - SOURCE_TABLE["energy"]
        )
    return row


def interleaved_plan(args) -> list[tuple[str, tuple]]:
    """Alternate the two methods, so that machine drift affects both equally.

    Running one method to completion and then the other would let a change in
    the machine -- another process starting, the clock stepping down -- land on
    one of them alone, and the whole point of the experiment is the ratio of the
    two times.
    """
    rfm = [
        ("rfm", (features, seed, repeat))
        for features in args.feature_counts
        for seed in args.seeds
        for repeat in range(args.repeats)
    ]
    weak = [
        ("wg", (subdivisions, WEAK_GALERKIN_EPSILON, repeat))
        for subdivisions in args.subdivisions
        for repeat in range(args.repeats)
    ]
    plan: list[tuple[str, tuple]] = []
    for index in range(max(len(rfm), len(weak))):
        if index < len(weak):
            plan.append(weak[index])
        if index < len(rfm):
            plan.append(rfm[index])
    return plan


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per configuration: the values, and the median of the repeats.

    The values themselves do not vary between repeats -- both methods are
    deterministic given their configuration -- so only the time is aggregated,
    and the spread of the repeats is reported beside it.
    """
    summary: list[dict[str, Any]] = []
    for method in ("weak Galerkin", "RFM"):
        block = [row for row in rows if row["method"] == method]
        key = "subdivisions" if method == "weak Galerkin" else "features"
        for size in sorted({row[key] for row in block}):
            group = [row for row in block if row[key] == size]
            times = np.asarray([row["wall_seconds"] for row in group])
            entry = {
                "method": method,
                key: size,
                "repeats": len(group),
                "lambda_abs_error_median": float(
                    np.median([row["lambda_abs_error"] for row in group])
                ),
                "energy_abs_error_median": float(
                    np.median([row["energy_abs_error"] for row in group])
                ),
                "wall_seconds_median": float(np.median(times)),
                "wall_seconds_min": float(times.min()),
                "wall_seconds_max": float(times.max()),
            }
            if method == "weak Galerkin":
                entry["unknowns"] = int(group[0]["unknowns"])
            else:
                entry["seeds"] = len({row["seed"] for row in group})
                entry["retained_median"] = float(
                    np.median([row["retained"] for row in group])
                )
            summary.append(entry)
    return summary


def check_source_agreement(rows: list[dict[str, Any]], factor: float) -> dict[str, Any]:
    """Whether the reimplementation reaches the source's printed accuracy.

    The gate is set before the run: at the finest mesh the source reports, the
    errors against the independent reference must be no worse than ``factor``
    times the errors the source's own printed values have.  It is a check on the
    reimplementation, not on the source.
    """
    finest = [
        row
        for row in rows
        if row["method"] == "weak Galerkin"
        and int(row["subdivisions"]) == SOURCE_TABLE["subdivisions"]
    ]
    if not finest:
        return {"checked": False}
    source_lambda = abs(SOURCE_TABLE["lambda"] - REFERENCE_LAMBDA)
    source_energy = abs(SOURCE_TABLE["energy"] - REFERENCE_ENERGY)
    ours_lambda = float(np.median([row["lambda_abs_error"] for row in finest]))
    ours_energy = float(np.median([row["energy_abs_error"] for row in finest]))
    return {
        "checked": True,
        "criterion": (
            f"errors at n={SOURCE_TABLE['subdivisions']} against the independent "
            f"reference within {factor} times the source's own"
        ),
        "source_lambda_error": source_lambda,
        "source_energy_error": source_energy,
        "reimplementation_lambda_error": ours_lambda,
        "reimplementation_energy_error": ours_energy,
        "passed": bool(
            ours_lambda <= factor * source_lambda
            and ours_energy <= factor * source_energy
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--feature-counts", type=int, nargs="+", default=list(DEFAULT_FEATURE_COUNTS)
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--subdivisions", type=int, nargs="+", default=list(DEFAULT_SUBDIVISIONS)
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--agreement-factor", type=float, default=1.05)
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args(argv)

    provenance.require_threads(args.threads)

    configuration = {
        "benchmark": BENCHMARK,
        "final_evaluation_order": FINAL_EVALUATION_ORDER,
        "reference": {"lambda": REFERENCE_LAMBDA, "energy": REFERENCE_ENERGY},
        "reference_provenance": "independent spectral solution, not either method",
        "feature_counts": list(args.feature_counts),
        "seeds": list(args.seeds),
        "feature_width": FEATURE_WIDTH,
        "center_spread": CENTER_SPREAD,
        "quadrature_order": QUADRATURE_ORDER,
        "evaluation_order": EVALUATION_ORDER,
        "mass_cutoff": MASS_CUTOFF,
        "subdivisions": list(args.subdivisions),
        "weak_galerkin_epsilon": WEAK_GALERKIN_EPSILON,
        "repeats": args.repeats,
        "threads": args.threads,
        "timing_scope": f"construction and solve, {args.threads} thread(s), methods interleaved",
    }
    run = provenance.open_run(
        EXPERIMENT, config=configuration, run_id=args.run_id, resume=args.resume
    )

    if not args.skip_warmup:
        print("warm-up (discarded)", flush=True)
        run_rfm_call(args.feature_counts[0], WARMUP_SEED)
        run_weak_galerkin_call(args.subdivisions[0], WEAK_GALERKIN_EPSILON)

    rows: list[dict[str, Any]] = []
    for method, parameters in interleaved_plan(args):
        if method == "rfm":
            features, seed, repeat = parameters
            call_id = f"rfm_N{features:05d}_seed{seed}_repeat{repeat:02d}"
        else:
            subdivisions, epsilon, repeat = parameters
            call_id = f"wg_n{subdivisions:04d}_eps{epsilon:g}_repeat{repeat:02d}"

        recorded = run.completed(call_id)
        if recorded is not None:
            rows.append(recorded)
            continue

        if method == "rfm":
            features, seed, repeat = parameters
            basis, result, seconds = run_rfm_call(features, seed)
            row = _rfm_row(features, seed, result, seconds, basis=basis)
        else:
            subdivisions, epsilon, repeat = parameters
            _, result, seconds = run_weak_galerkin_call(subdivisions, epsilon)
            row = _weak_galerkin_row(subdivisions, epsilon, result, seconds)
        row["repeat"] = repeat
        rows.append(run.complete(call_id, row))
        print(
            f"{row['method']:14s} "
            f"{row.get('features', row.get('subdivisions')):6d} "
            f"repeat {repeat}  lambda error {row['lambda_abs_error']:.3e}  "
            f"{row['wall_seconds']:.2f}s",
            flush=True,
        )

    provenance.write_csv(run.path("trials.csv"), rows)
    provenance.write_csv(run.path("summary.csv"), summarize(rows))
    run.seal(
        reference_lambda=REFERENCE_LAMBDA,
        reference_energy=REFERENCE_ENERGY,
        source_agreement=check_source_agreement(rows, args.agreement_factor),
    )
    print(f"written to {run.directory}")


if __name__ == "__main__":
    main()
