r"""Example 6 (Section 4.2): a rotating two-component dipolar condensate.

The hardest problem of Section 4: two coupled complex components, a rotation
term, and a nonlocal dipolar interaction.  The comparison is against the
gradient flow method of Li et al. at the two mesh widths their table reports.

Three points of protocol carry most of the work, and each exists to remove a
choice that would otherwise decide the answer.

*Which initial state.*  A gradient flow converges to whatever basin it starts
in, and the source prescribes a search over the ordered product of ten initial
formulas for each of the two components -- a hundred starts.  That search is run
once here, on the coarsest mesh, and the best few candidates are re-solved on the
next mesh and must keep their ordering.  The ground state it identifies is what
every later run starts from.

*How long the baseline takes.*  Across the starts that all reach the same ground
state, the iteration count varies by nearly a factor of eight, so timing one of
them reports an arbitrary draw.  What is reported is the median over a fixed
sample of the convergent starts, which is the counterpart of the median over
feature draws on the other side.  The sample is taken at even spacing in the
source's own product order, fixed before any timing is observed.

*What the reference is.*  Li et al. build theirs at a mesh width of one
sixteenth; their own table shows the quarter-width energy already within
:math:`9\times10^{-14}` of it, well below the errors under comparison, so
the quarter-width solution is recomputed here and used as the reference.  It is
still checked against their printed values before anything is measured against
it.

Run with::

    python -m rfmeig.experiments.exp6_dipolar --run-id my-run
"""

from __future__ import annotations

import argparse
import itertools
import time
from typing import Any

import numpy as np

from rfmeig import provenance
from rfmeig.baselines.gflm_ktm import (
    PUBLISHED_INITIAL_KINDS,
    GFLMKTMSettings,
)
from rfmeig.baselines.gflm_ktm_threaded import (
    ThreadedGFLMKTMSystem,
    solve_gflm_ktm_threaded,
)
from rfmeig.box_basis import BoxFeatureBasis
from rfmeig.final_evaluation import dipolar_fields_on_reference_nodes
from rfmeig.problems.dipolar import (
    DipolarBenchmark,
    DipolarSettings,
    evaluate_dipolar_feature_state,
)
from rfmeig.problems.dipolar_fast import solve_fast

EXPERIMENT = "experiment6"

#: The two mesh widths that are compared.  A quarter is the reference and is not
#: itself a row of the table.
COMPARISON_WIDTHS = (1.0, 0.5)
REFERENCE_WIDTH = 0.25
#: The search runs on the coarsest mesh; the best few are confirmed on the next.
SEARCH_WIDTH = 1.0
CONFIRMATION_WIDTH = 0.5
CONFIRMATION_TOP_K = 3

#: Four feature counts, spanning a factor of six and three orders of error.
FEATURE_COUNTS = (256, 512, 1024, 1536)
#: The ten reporting draws.
SEEDS = tuple(2086156109 + 42 * index for index in range(10))
#: How many times each draw is solved for the clock.  The solve is deterministic
#: given the draw, so the repeats change no reported value; they exist so that a
#: configuration's time is a median rather than one sample of a busy machine.
TIMED_REPEATS = 3
WARMUP_SEED = 1086156001

#: How many convergent starts are timed, and the least that will be accepted.
#: A median does not need the whole population: over the starts measured here it
#: moves by under five percent between a sample of twenty and all of them, far
#: below the eightfold spread the sample exists to summarize.
TIMED_INITIAL_PAIRS = 20
MINIMUM_TIMED_INITIAL_PAIRS = 10
#: Two starts are in the same basin when their energies agree far more closely
#: than the discretization error being measured: the fixed-point tolerance is
#: 1e-11 and the coarsest reported error is about 6e-4.
BASIN_ENERGY_TOLERANCE = 1.0e-8

#: Li et al., Table 2, at each mesh width: the error of the energy, of each
#: chemical potential, and the virial residual.  The last is not an error against
#: a reference at all -- it is the amount by which the converged state fails an
#: identity every exact solution satisfies -- which is why it can be reported
#: without one.  The reproduction is required to land within
#: ``ALIGNMENT_TOLERANCE`` of every one of the four before anything is compared.
PUBLISHED_TABLE = {
    1.0: (6.1494e-4, 5.4000e-3, 5.7000e-3, 2.9300e-2),
    0.5: (6.9304e-7, 3.4110e-6, 3.5881e-6, 2.8138e-5),
    0.25: (9.0594e-14, 5.6914e-12, 5.9934e-12, 1.9971e-11),
}
#: The realized gaps are about 0.18 percent at unit width and 0.012 percent at
#: half width, so this bar sits a factor of three above what is achieved and far
#: below any error being compared.
ALIGNMENT_TOLERANCE = 5.0e-3
#: The source's own printed reference, to the digits it prints.
PUBLISHED_REFERENCE = {"energy": 8.3727, "mu1": 12.2977, "mu2": 12.4845}

#: The feature law, fixed in advance.  Four widths rather than one, because the
#: state has both a broad envelope and a vortex core.
FEATURE_WIDTHS = (0.035, 0.05, 0.065, 0.08)
CENTER_SPREAD = 0.18
GRID_SIZE = 128
MASS_CUTOFF = 1.0e-12
MAX_ITERATIONS = 1400
RESIDUAL_TOLERANCE = 1.0e-8
PADDING_FACTOR = 3
#: Separates the stream that perturbs the initial state from the one that draws
#: the features, so that neither is a function of the other.
INITIAL_SEED_OFFSET = 1543


def initial_pairs() -> tuple[tuple[str, str], ...]:
    """The source's ordered product of the ten initial formulas, for two components."""
    return tuple(itertools.product(PUBLISHED_INITIAL_KINDS, repeat=2))


def sampled_pair_indices(count: int, total: int) -> tuple[int, ...]:
    """``count`` starts, evenly spaced in the source's product order.

    Even spacing in an order that was fixed before any timing was observed, and
    that has nothing to do with how fast a start converges, is what keeps the
    sample from being a selection.
    """
    if count >= total:
        return tuple(range(total))
    return tuple(int(round(index)) for index in np.linspace(0, total - 1, count))


def rfm_settings() -> DipolarSettings:
    return DipolarSettings(
        grid_size=GRID_SIZE,
        mass_cutoff=MASS_CUTOFF,
        max_iterations=MAX_ITERATIONS,
        residual_tolerance=RESIDUAL_TOLERANCE,
        padding_factor=PADDING_FACTOR,
    )


def make_basis(features: int, seed: int, benchmark: DipolarBenchmark):
    return BoxFeatureBasis.sample_multiscale_rbf(
        features, FEATURE_WIDTHS, seed, benchmark.bounds, CENTER_SPREAD
    )


def phase_free_error(reference: np.ndarray, approximate: np.ndarray) -> float:
    r"""Li et al. (4.2): :math:`\max|\phi-\phi^h|/\max|\phi|`, phase removed.

    The energy depends on the components only through their densities, so each
    carries an arbitrary global phase.  The phase that minimizes the L2 difference
    is divided out before the maximum is taken; without that step the quantity
    measures whatever phase the solver happened to return.
    """
    reference = np.asarray(reference).reshape(-1)
    approximate = np.asarray(approximate).reshape(-1)
    if reference.shape != approximate.shape:
        raise ValueError("the two fields must live on the same nodes")
    overlap = np.vdot(reference, approximate)
    magnitude = abs(overlap)
    phase = np.conjugate(overlap) / magnitude if magnitude > 0.0 else 1.0 + 0.0j
    scale = float(np.max(np.abs(reference)))
    if scale <= 0.0:
        raise ValueError("the reference field vanishes identically")
    return float(np.max(np.abs(reference - approximate * phase))) / scale


def restriction_stride(mesh_width: float) -> int:
    """A coarse node is a reference node, so restriction is exact."""
    ratio = mesh_width / REFERENCE_WIDTH
    stride = int(round(ratio))
    if not np.isclose(ratio, stride) or stride < 1:
        raise ValueError(f"h={mesh_width:g} does not refine the reference mesh")
    return stride


def gflm_settings(mesh_width: float, benchmark: DipolarBenchmark) -> GFLMKTMSettings:
    """The baseline at one mesh width, with the source's own time step and tolerance."""
    lower, upper = benchmark.bounds[0]
    requested = (upper - lower) / mesh_width
    if not np.isclose(requested, round(requested)):
        raise ValueError("the mesh width must divide the box")
    return GFLMKTMSettings(mesh_width=float(mesh_width))


# --------------------------------------------------------------------------
# the search for the ground state
# --------------------------------------------------------------------------
def search_ground_state(benchmark: DipolarBenchmark, *, workers: int) -> dict[str, Any]:
    """Solve the full product on the coarse mesh, then confirm the best few.

    Returns the index of the start that reaches the lowest energy, the indices
    of every start that reaches the same basin, and the confirmation record.
    A start that does not meet the residual tolerance is not eligible, and if
    none is, the run stops rather than reporting the least bad one.
    """
    pairs = initial_pairs()
    settings = gflm_settings(SEARCH_WIDTH, benchmark)
    system = ThreadedGFLMKTMSystem(benchmark, settings, workers=workers)

    energies: list[float | None] = []
    for index, pair in enumerate(pairs):
        result = solve_gflm_ktm_threaded(
            benchmark,
            settings,
            initial_kinds=pair,
            workers=workers,
            capture_history=False,
            system=system,
        )
        energies.append(float(result.energy) if result.converged else None)
        if (index + 1) % 20 == 0:
            print(f"  search: {index + 1}/{len(pairs)} starts", flush=True)

    eligible = [i for i, energy in enumerate(energies) if energy is not None]
    if not eligible:
        raise RuntimeError(
            "no prescribed initial state met the fixed-point residual tolerance"
        )
    best = min(eligible, key=lambda i: energies[i])
    lowest = energies[best]
    basin = [i for i in eligible if abs(energies[i] - lowest) <= BASIN_ENERGY_TOLERANCE]

    # Confirm on the next mesh: the ordering of the best few must not change.
    ranked = sorted(eligible, key=lambda i: energies[i])[:CONFIRMATION_TOP_K]
    finer = gflm_settings(CONFIRMATION_WIDTH, benchmark)
    confirmation = []
    for index in ranked:
        result = solve_gflm_ktm_threaded(
            benchmark,
            finer,
            initial_kinds=pairs[index],
            workers=workers,
            capture_history=False,
        )
        confirmation.append(
            {
                "pair_index": index,
                "pair": list(pairs[index]),
                "coarse_energy": energies[index],
                "confirmed_energy": float(result.energy),
                "converged": bool(result.converged),
            }
        )
    ordering_kept = [row["confirmed_energy"] for row in confirmation] == sorted(
        row["confirmed_energy"] for row in confirmation
    )
    return {
        "best_pair_index": best,
        "best_pair": list(pairs[best]),
        "lowest_energy": lowest,
        "basin_pair_indices": basin,
        "eligible_starts": len(eligible),
        "total_starts": len(pairs),
        "confirmation": confirmation,
        "confirmation_ordering_kept": ordering_kept,
    }


def check_published_alignment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Whether the reimplementation lands on the source's printed table.

    Checked before anything is compared against it: a baseline that does not
    reproduce its own source's numbers is not a baseline.
    """
    gaps: dict[str, float] = {}
    passed = True
    for width in sorted({float(row["mesh_width"]) for row in rows}):
        if width not in PUBLISHED_TABLE:
            continue
        # One mesh width is solved from several starts; they all reach the same
        # state, so the median is the representative and the spread is a check.
        block = [row for row in rows if float(row["mesh_width"]) == width]
        ours = (
            float(np.median([row["energy_abs_error"] for row in block])),
            float(np.median([row["mu1_abs_error"] for row in block])),
            float(np.median([row["mu2_abs_error"] for row in block])),
            float(np.median([row["virial_residual"] for row in block])),
        )
        for name, expected, found in zip(
            ("energy", "mu1", "mu2", "virial"),
            PUBLISHED_TABLE[width],
            ours,
            strict=True,
        ):
            gap = abs(found - expected) / expected
            gaps[f"h{width:g}_{name}"] = gap
            passed = passed and gap <= ALIGNMENT_TOLERANCE
    if not gaps:
        # Nothing was checked; saying "passed" here would report a check that
        # never ran.
        return {"tolerance": ALIGNMENT_TOLERANCE, "checked": False, "relative_gaps": {}}
    return {
        "tolerance": ALIGNMENT_TOLERANCE,
        "checked": True,
        "passed": passed,
        "relative_gaps": gaps,
    }


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------
def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per configuration; the baseline's time is a median over starts."""
    summary: list[dict[str, Any]] = []
    for method, key in (("GFLM-KTM", "mesh_width"), ("RFM", "features")):
        block = [row for row in rows if row["method"] == method]
        for size in sorted({row[key] for row in block}):
            group = [row for row in block if row[key] == size]
            times = np.asarray([row["wall_seconds"] for row in group])
            summary.append(
                {
                    "method": method,
                    key: size,
                    "samples": len(group),
                    "energy_abs_error": float(
                        np.median([row["energy_abs_error"] for row in group])
                    ),
                    "mu1_abs_error": float(
                        np.median([row["mu1_abs_error"] for row in group])
                    ),
                    "mu2_abs_error": float(
                        np.median([row["mu2_abs_error"] for row in group])
                    ),
                    "wave_error": float(
                        np.median(
                            [
                                max(row["wave_error_1"], row["wave_error_2"])
                                for row in group
                            ]
                        )
                    ),
                    "virial_residual": float(
                        np.median([row["virial_residual"] for row in group])
                    ),
                    "wall_seconds_median": float(np.median(times)),
                    "wall_seconds_min": float(times.min()),
                    "wall_seconds_max": float(times.max()),
                }
            )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--feature-counts", type=int, nargs="+", default=list(FEATURE_COUNTS)
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--mesh-widths", type=float, nargs="+", default=list(COMPARISON_WIDTHS)
    )
    parser.add_argument("--timed-pairs", type=int, default=TIMED_INITIAL_PAIRS)
    parser.add_argument(
        "--repeats",
        type=int,
        default=TIMED_REPEATS,
        help="how many times each feature draw is solved for the clock; the "
        "reported values do not depend on it, only the timing does",
    )
    parser.add_argument(
        "--sides",
        choices=("both", "rfm", "baseline"),
        default="both",
        help="which side of the comparison to run; the reference is computed "
        "either way, since both sides are measured against it",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="reuse the ground state recorded by an earlier run of this driver",
    )
    parser.add_argument("--best-pair-index", type=int, default=None)
    args = parser.parse_args(argv)

    provenance.require_threads(args.threads)
    benchmark = DipolarBenchmark()

    configuration = {
        "benchmark": "two-component rotating dipolar condensate",
        "mesh_widths": list(args.mesh_widths),
        "reference_mesh_width": REFERENCE_WIDTH,
        "feature_counts": list(args.feature_counts),
        "seeds": list(args.seeds),
        "timed_repeats": args.repeats,
        "feature_widths": list(FEATURE_WIDTHS),
        "center_spread": CENTER_SPREAD,
        "grid_size": GRID_SIZE,
        "field_evaluation_nodes": "GFLM-KTM left-endpoint reference nodes",
        "timed_initial_pairs": args.timed_pairs,
        "initial_state_rule": (
            "ordered product of the source's ten formulas for two components; "
            "lowest energy among the starts meeting the residual tolerance"
        ),
        "baseline_timing_rule": (
            "median over evenly spaced convergent starts, not a single start"
        ),
        "threads": args.threads,
        "workers": args.workers,
        "sides": args.sides,
        "best_pair_index": args.best_pair_index,
    }
    run = provenance.open_run(
        EXPERIMENT, config=configuration, run_id=args.run_id, resume=args.resume
    )

    # ---- which initial state ---------------------------------------------
    if args.best_pair_index is not None:
        search = {"best_pair_index": args.best_pair_index, "source": "given"}
    else:
        recorded = run.completed("search")
        if recorded is not None:
            search = recorded
        else:
            print("searching the prescribed initial states", flush=True)
            search = run.complete(
                "search", search_ground_state(benchmark, workers=args.workers)
            )
    pairs = initial_pairs()
    best_pair = pairs[int(search["best_pair_index"])]
    print(f"ground state from initial pair {best_pair}", flush=True)

    # ---- the reference ----------------------------------------------------
    reference_settings = gflm_settings(REFERENCE_WIDTH, benchmark)
    lower, upper = benchmark.bounds[0]
    reference_grid_size = int(round((upper - lower) / REFERENCE_WIDTH))
    print(
        f"reference at h={REFERENCE_WIDTH:g} ({reference_grid_size} nodes a side)",
        flush=True,
    )
    reference = solve_gflm_ktm_threaded(
        benchmark,
        reference_settings,
        initial_kinds=best_pair,
        workers=args.workers,
        capture_history=False,
    )
    if not reference.converged:
        raise RuntimeError("the reference solve did not converge")
    reference_states = reference.states
    reference_mu = reference.chemical_potentials
    rounded = {
        "energy": round(float(reference.energy), 4),
        "mu1": round(float(reference_mu[0]), 4),
        "mu2": round(float(reference_mu[1]), 4),
    }
    if rounded != PUBLISHED_REFERENCE:
        raise RuntimeError(
            f"the reference does not round to the source's printed values: "
            f"{rounded} against {PUBLISHED_REFERENCE}"
        )
    print(f"  reference reproduces the source's printed values: {rounded}", flush=True)

    rows: list[dict[str, Any]] = []

    def measured(result, mesh_width: float | None, *, fields=None) -> dict[str, float]:
        """Errors of one solved state against the reference, on the coarse nodes.

        A coarse mesh is a subset of the reference mesh, so the reference is
        restricted by taking every ``stride`` th node rather than interpolated;
        the comparison is then exact at the nodes and carries no interpolation
        error of its own.  ``mesh_width`` of ``None`` means the state was already
        evaluated on the reference's own grid, as the random feature states are.
        """
        stride = 1 if mesh_width is None else restriction_stride(mesh_width)
        states = result.states if fields is None else fields
        mu = result.chemical_potentials
        return {
            "energy": float(result.energy),
            "mu1": float(mu[0]),
            "mu2": float(mu[1]),
            "energy_abs_error": abs(float(result.energy) - float(reference.energy)),
            "mu1_abs_error": abs(float(mu[0]) - float(reference_mu[0])),
            "mu2_abs_error": abs(float(mu[1]) - float(reference_mu[1])),
            "wave_error_1": phase_free_error(
                reference_states[0][::stride, ::stride], states[0]
            ),
            "wave_error_2": phase_free_error(
                reference_states[1][::stride, ::stride], states[1]
            ),
            "virial_residual": float(getattr(result, "virial_residual", np.nan)),
        }

    # ---- the baseline, timed over a sample of convergent starts -----------
    # Either side can be run alone.  The reference above is computed either way,
    # because both sides are measured against it.
    basin = search.get("basin_pair_indices") or [int(search["best_pair_index"])]
    timed = sampled_pair_indices(args.timed_pairs, len(basin))
    selected = [basin[index] for index in timed]
    if (
        len(selected) < MINIMUM_TIMED_INITIAL_PAIRS
        and len(basin) >= MINIMUM_TIMED_INITIAL_PAIRS
    ):
        raise RuntimeError("too few starts selected for a meaningful median")

    for mesh_width in args.mesh_widths if args.sides != "rfm" else ():
        settings = gflm_settings(mesh_width, benchmark)
        for pair_index in selected:
            call_id = f"gflm_h{mesh_width:g}_pair{pair_index:03d}"
            recorded = run.completed(call_id)
            if recorded is not None:
                rows.append(recorded)
                continue
            started = time.perf_counter()
            result = solve_gflm_ktm_threaded(
                benchmark,
                settings,
                initial_kinds=pairs[pair_index],
                workers=args.workers,
                capture_history=False,
            )
            seconds = time.perf_counter() - started
            row = {
                "method": "GFLM-KTM",
                "mesh_width": mesh_width,
                "pair_index": pair_index,
                "pair": " + ".join(pairs[pair_index]),
                "iterations": int(result.iterations),
                "converged": bool(result.converged),
                "wall_seconds": seconds,
                **measured(result, mesh_width),
            }
            rows.append(run.complete(call_id, row))
            print(
                f"GFLM h={mesh_width:g} pair {pair_index:3d}: "
                f"energy error {row['energy_abs_error']:.3e}  {seconds:.1f}s",
                flush=True,
            )

    alignment = check_published_alignment(
        [row for row in rows if row["method"] == "GFLM-KTM"]
    )
    if not alignment["checked"]:
        print("published alignment: not checked (the baseline was not run)", flush=True)
    else:
        print(
            f"published alignment: {'passed' if alignment['passed'] else 'FAILED'} "
            f"(tolerance {ALIGNMENT_TOLERANCE:g})",
            flush=True,
        )
    for name, gap in sorted(alignment["relative_gaps"].items()):
        marker = " " if gap <= ALIGNMENT_TOLERANCE else "*"
        print(f"  {marker} {name:20s} {gap:.3e}", flush=True)

    # ---- the random feature method ----------------------------------------
    settings = rfm_settings()
    plan = (
        [
            (features, seed, repeat)
            for features in args.feature_counts
            for seed in args.seeds
            for repeat in range(args.repeats)
        ]
        if args.sides != "baseline"
        else []
    )
    for features, seed, repeat in plan:
        call_id = f"rfm_N{features:05d}_seed{seed}_repeat{repeat:02d}"
        recorded = run.completed(call_id)
        if recorded is not None:
            rows.append(recorded)
            continue
        started = time.perf_counter()
        basis = make_basis(features, seed, benchmark)
        # The optimizer starts from the lowest mode of the linear part,
        # perturbed by a little noise: without the perturbation both
        # components would start identical and the rotation could not break
        # the symmetry between them.  The offset keeps that noise from
        # reusing the stream the features were drawn from.
        result = solve_fast(
            benchmark, basis, settings, initial_seed=int(seed) + INITIAL_SEED_OFFSET
        )
        seconds = time.perf_counter() - started
        # Outside the clock: the state is put on the reference's own grid,
        # so that both methods are compared at the same nodes.
        grid = evaluate_dipolar_feature_state(
            benchmark,
            basis,
            result.feature_coefficients,
            grid_size=reference_grid_size,
            padding_factor=PADDING_FACTOR,
        )
        row = {
            "method": "RFM",
            "features": features,
            "seed": seed,
            "repeat": repeat,
            "retained": int(result.retained_rank),
            "iterations": int(result.iterations),
            "converged": bool(result.converged),
            "wall_seconds": seconds,
            **measured(grid, None, fields=dipolar_fields_on_reference_nodes(
                benchmark, basis, result.feature_coefficients, reference_states[0].shape
            )),
        }
        rows.append(run.complete(call_id, row))
        print(
            f"RFM N={features:5d} seed {seed} repeat {repeat}: "
            f"energy error {row['energy_abs_error']:.3e}  {seconds:.1f}s",
            flush=True,
        )

    provenance.write_csv(run.path("trials.csv"), rows)
    provenance.write_csv(run.path("summary.csv"), summarize(rows))
    run.seal(
        reference_energy=float(reference.energy),
        reference_mu1=float(reference_mu[0]),
        reference_mu2=float(reference_mu[1]),
        best_initial_pair=list(best_pair),
        published_alignment=alignment,
    )
    print(f"written to {run.directory}")


if __name__ == "__main__":
    main()
