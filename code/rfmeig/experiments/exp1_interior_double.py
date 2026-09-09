r"""Example 1 (Section 4.1): an interior double eigenvalue on the square.

The operator is :math:`-\Delta+V(x)+V(y)` on :math:`(0,1)^2` with the
double-kink potential of :mod:`rfmeig.problems.separable`.  Because the
potential is separable, the exact spectrum is the set of sums of one-dimensional
levels, and the sum of the second and third one-dimensional levels is attained
by two different ordered pairs.  It is therefore a double eigenvalue, it sits in
the interior of the spectrum with a level below it and a level above it, and its
value is known to the accuracy of a one-dimensional solve.  This is the
configuration the rates of Section 3 are stated for, and the purpose of the
experiment is to measure them.

Four quantities are recorded for every draw:

``h1_gap``
    the :math:`H^1` gap between the target eigenspace and the two Ritz functions
    that approximate it, which is the quantity Theorem 3.9 bounds;
``eta_f_h1``
    how well the retained trial space approximates the lower spectral subspace
    through the target level, which is the quantity that enters that bound;
``cluster_relative_error``
    the largest relative error among the two computed levels;
``splitting_relative_error``
    how far the two computed levels have drifted apart, which a method that did
    not respect the multiplicity would show as a spurious splitting.

The two external gaps are recorded as well.  They decide whether the cluster has
been identified at all, and a draw in which either of them fails to be positive
is reported rather than discarded.

Run with::

    python -m rfmeig.experiments.exp1_interior_double --run-id my-run
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from rfmeig import features, metrics, provenance
from rfmeig.problems.separable import (
    DoubleKinkPotential,
    leading_pairs,
    locate_pair,
    solve_one_dimensional,
)
from rfmeig.rayleigh_ritz import (
    pencil_residuals,
    solve_energy_whitened,
)
from rfmeig.separable_assembly import SeparableSquareQuadrature

EXPERIMENT = "experiment1"

#: The one-dimensional levels whose sum is the target, zero-based.
TARGET_PAIR = (1, 2)
#: How many one-dimensional levels are enumerated when the spectrum is ordered.
PAIR_LIMIT = 18
#: How many Ritz values are computed; enough to place the target and its
#: neighbours without solving for the whole reduced spectrum.
RITZ_COUNT = 10

DEFAULT_FEATURE_COUNTS = (128, 160, 216, 288, 384, 512, 672, 896)
DEFAULT_DRAWS = 100
#: The base seed of the recorded run; the seed of one draw is
#: ``base + 137 * N + draw``, so that no two configurations share a realization.
DEFAULT_BASE_SEED = 2031080000
DEFAULT_TAIL_EXPONENT = 2.42
DEFAULT_FREQUENCY_SCALE = 8.5
DEFAULT_ASSEMBLY_ORDER = 512
DEFAULT_EVALUATION_ORDER = 512
DEFAULT_SUBSPACE_ORDER = 512
DEFAULT_REFERENCE_MODES = 112
DEFAULT_REFERENCE_ORDER = 96
DEFAULT_TRUNCATION = 1.0e-12
DEFAULT_REFERENCE_VALUE = 149.91769837259983

#: Means are the manuscript statistic; quantiles are retained as diagnostics.
RATE_FIELDS = ("h1_gap", "eta_f_h1", "cluster_relative_error")
RATE_LEVEL = 0.9


def draw_seed(base: int, count: int, draw: int) -> int:
    """The seed of one draw, from the base seed, the feature count and the index."""
    return base + 137 * count + draw


def run_draw(
    *, feature_count: int, seed: int, potential: DoubleKinkPotential,
    tail_exponent: float, frequency_scale: float, truncation: float,
    assembly: SeparableSquareQuadrature, evaluation: SeparableSquareQuadrature,
    subspace: SeparableSquareQuadrature, first: int, last: int, exact: float,
) -> dict[str, Any]:
    """Draw, assemble and solve, then measure on the retained space.

    All three rules are composite. When their orders coincide, their matrices
    are shared; the tensor rule itself is unchanged by this factorization.
    """
    started = time.perf_counter()
    omega, phase = features.sample_inverse_cdf_features(
        feature_count, tail_exponent, seed, frequency_scale
    )
    sample_seconds = time.perf_counter() - started
    started = time.perf_counter()
    forms = assembly.assemble(omega, phase)
    assemble_seconds = time.perf_counter() - started
    started = time.perf_counter()
    solution = solve_energy_whitened(forms.energy, forms.mass, RITZ_COUNT, tolerance=truncation)
    solve_seconds = time.perf_counter() - started
    residuals = pencil_residuals(forms.energy, forms.mass, solution.eigenvalues, solution.coefficients)
    started = time.perf_counter()
    target_forms = forms if evaluation is assembly else evaluation.assemble(omega, phase)
    target = solution.coefficients[:, first - 1:last]
    gap = metrics.eigenspace_gap_from_grams(
        target_forms.reference_gram[-2:, -2:], target.T @ target_forms.h1 @ target,
        target_forms.cross[-2:] @ target,
    )
    target_seconds = time.perf_counter() - started
    started = time.perf_counter()
    subspace_forms = (forms if subspace is assembly else target_forms if subspace is evaluation
                      else subspace.assemble(omega, phase))
    basis = solution.basis
    approximation = metrics.best_approximation_from_grams(
        subspace_forms.reference_gram[:last, :last], basis.T @ subspace_forms.h1 @ basis,
        subspace_forms.cross[:last] @ basis, relative_tolerance=1.0e-13,
    )
    row = {
        "N": feature_count, "seed": seed, "retained": solution.retained,
        "h1_gap": gap, "eta_f_h1": approximation,
        **metrics.cluster_errors(solution.eigenvalues, first, last, exact),
        "pencil_residual_max": float(np.max(residuals)),
        "sample_s": sample_seconds, "assemble_s": assemble_seconds,
        "eigensolve_s": solve_seconds, "target_evaluation_s": target_seconds,
        "subspace_evaluation_s": time.perf_counter() - started,
        "method_s": sample_seconds + assemble_seconds + solve_seconds,
    }
    row["index_success"] = int(row["lower_external_gap_relative"] > 0.0
                               and row["upper_external_gap_relative"] > 0.0)
    row.update({f"lambda_{i}": float(v) for i, v in enumerate(solution.eigenvalues, start=1)})
    row.update({f"truncation_{key}": value for key, value in solution.diagnostics.items()})
    return row


def summarize(rows: list[dict[str, Any]], counts: list[int]) -> list[dict[str, Any]]:
    """Arithmetic mean, median and upper quantile at each feature count."""
    summary: list[dict[str, Any]] = []
    for count in counts:
        block = [row for row in rows if int(row["N"]) == count]
        entry: dict[str, Any] = {
            "N": count,
            "draws": len(block),
            "samples": len(block),
            "retained_median": float(np.median([row["retained"] for row in block])),
            "index_success_count": int(sum(row["index_success"] for row in block)),
        }
        for field in (
            "h1_gap",
            "eta_f_h1",
            "cluster_relative_error",
            "splitting_relative_error",
            "lower_external_gap_relative",
            "upper_external_gap_relative",
            "method_s",
        ):
            sample = np.asarray([float(row[field]) for row in block])
            entry[f"{field}_mean"] = float(np.mean(sample))
            entry[f"{field}_median"] = float(np.median(sample))
            entry[f"{field}_q90"] = float(np.quantile(sample, RATE_LEVEL))
        summary.append(entry)
    return summary


def fit_rates(rows: list[dict[str, Any]], counts: list[int]) -> dict[str, Any]:
    """Fit all budgets, with stratified bootstrap intervals for the means."""
    if len(counts) < 2:
        return {}
    rates = {}
    for field in RATE_FIELDS:
        for name, reducer in (("mean", np.mean), ("q90", lambda x: np.quantile(x, RATE_LEVEL))):
            values = [float(reducer([row[field] for row in rows if int(row["N"]) == n]))
                      for n in counts]
            rates[f"{field}_{name}_slope"] = metrics.regression_slope(counts, values)
        rates[f"{field}_mean_slope_bootstrap_95"] = list(metrics.bootstrap_slope_interval(
            rows, counts, field, statistic="mean"
        ))
    return rates


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default=None, help="name of the output directory")
    parser.add_argument(
        "--resume", action="store_true", help="continue an unsealed run"
    )
    parser.add_argument(
        "--feature-counts", type=int, nargs="+", default=list(DEFAULT_FEATURE_COUNTS)
    )
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--tail-exponent", type=float, default=DEFAULT_TAIL_EXPONENT)
    parser.add_argument(
        "--frequency-scale", type=float, default=DEFAULT_FREQUENCY_SCALE
    )
    parser.add_argument("--assembly-order", type=int, default=DEFAULT_ASSEMBLY_ORDER)
    parser.add_argument(
        "--evaluation-order", type=int, default=DEFAULT_EVALUATION_ORDER
    )
    parser.add_argument("--subspace-order", type=int, default=DEFAULT_SUBSPACE_ORDER)
    parser.add_argument("--reference-modes", type=int, default=DEFAULT_REFERENCE_MODES)
    parser.add_argument("--reference-order", type=int, default=DEFAULT_REFERENCE_ORDER)
    parser.add_argument("--reference-value", type=float, default=DEFAULT_REFERENCE_VALUE)
    parser.add_argument("--truncation", type=float, default=DEFAULT_TRUNCATION)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)

    if args.draws < 1 or len(set(args.feature_counts)) != len(args.feature_counts):
        parser.error("draws must be positive and feature counts must be distinct")
    provenance.require_threads(args.threads)
    potential = DoubleKinkPotential()

    configuration = {
        "potential": potential.describe(),
        "reaction": "c(x,y)=V(x)+V(y)",
        "target_pair_zero_based": list(TARGET_PAIR),
        "feature_counts": list(args.feature_counts),
        "draws": args.draws,
        "seed_rule": f"{args.base_seed}+137*N+draw",
        "tail_exponent": args.tail_exponent,
        "frequency_scale": args.frequency_scale,
        "assembly_composite_order_per_cell": args.assembly_order,
        "evaluation_composite_order_per_interval": args.evaluation_order,
        "subspace_composite_order_per_interval": args.subspace_order,
        "reference_modes": args.reference_modes,
        "reference_composite_order_per_cell": args.reference_order,
        "truncation_tolerance": args.truncation,
        "ritz_count": RITZ_COUNT,
        "reference_value": args.reference_value,
        "statistic": "arithmetic mean",
        "quadrature": "separated evaluation of composite tensor Gauss",
    }
    run = provenance.open_run(
        EXPERIMENT, config=configuration, run_id=args.run_id, resume=args.resume
    )

    reference_values, reference_vectors = solve_one_dimensional(
        potential, modes=args.reference_modes, order=args.reference_order
    )
    first, last, exact = locate_pair(reference_values, TARGET_PAIR, limit=PAIR_LIMIT)
    if first == last:
        raise RuntimeError(
            "the target eigenvalue is simple; the experiment needs a double one"
        )

    pairs = leading_pairs(reference_values, last, limit=PAIR_LIMIT) + [
        TARGET_PAIR, tuple(reversed(TARGET_PAIR))
    ]
    rules = {order: SeparableSquareQuadrature(potential, reference_vectors, pairs, order)
             for order in {args.assembly_order, args.evaluation_order, args.subspace_order}}
    assembly, evaluation, subspace = (rules[order] for order in (
        args.assembly_order, args.evaluation_order, args.subspace_order
    ))
    exact = args.reference_value

    rows: list[dict[str, Any]] = []
    for count in args.feature_counts:
        for draw in range(args.draws):
            call_id = f"N{count:05d}_draw{draw:04d}"
            recorded = run.completed(call_id)
            if recorded is not None:
                rows.append(recorded)
                continue
            row = run_draw(
                feature_count=count,
                seed=draw_seed(args.base_seed, count, draw),
                potential=potential,
                tail_exponent=args.tail_exponent,
                frequency_scale=args.frequency_scale,
                truncation=args.truncation,
                assembly=assembly,
                evaluation=evaluation,
                subspace=subspace,
                first=first,
                last=last,
                exact=exact,
            )
            row["draw"] = draw
            row["sample"] = draw
            rows.append(run.complete(call_id, row))
        print(f"N={count}: {args.draws} draws complete", flush=True)

    summary = summarize(rows, list(args.feature_counts))
    provenance.write_csv(run.path("trials.csv"), rows)
    provenance.write_csv(run.path("summary.csv"), summary)
    run.seal(
        exact_eigenvalue=exact,
        target_positions_one_based=[first, last],
        **fit_rates(rows, list(args.feature_counts)),
    )
    print(f"written to {run.directory}")


if __name__ == "__main__":
    main()
