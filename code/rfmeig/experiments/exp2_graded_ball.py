r"""Example 2: the radially graded unit ball, a(r) = 1 + 3 r^2.

Compare the first ten eigenvalues using conforming random features,
isoparametric P2 finite elements, and Bernstein Eig-PIELM collocation.
The reference spectrum is computed independently from the radial equations.

The RFM frequency scale is selected by agreement between two independent
feature draws. Every configuration records errors, timed repetitions, and
identification of the multiple levels. Completed configurations can be resumed.

    RFMEIG_THREADS=6 python -m rfmeig.experiments.exp2_graded_ball --run-id my-run
"""

from __future__ import annotations

import argparse
import functools
import time
import warnings
from typing import Any

import numpy as np

from rfmeig import provenance
from rfmeig.baselines import eig_pielm, isoparametric_fem
from rfmeig.problems.graded_ball import GradedBall
from rfmeig.rfm_ball import solve as rfm_solve

#: The ten ordered levels span 1 + 3 + 1 + 5, so ten is the smallest count that
#: closes both multiple levels; nine would cut the fivefold cluster in half.
DEFAULT_COUNT = 10

#: Twenty draws is what the manuscript's captions state, and it is enough for
#: the observed range to mean something at the smallest budget, where the
#: draw-to-draw spread is widest.
DEFAULT_DRAWS = 20

#: Each configuration is solved three times and the median clock kept, so that
#: one unlucky scheduling slice does not become the reported time.
DEFAULT_REPEATS = 3

#: Candidate frequency scales for the blind selection rule.
DEFAULT_SCALES = (2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 14.0)

#: (features, quadrature level) for the sampled space.  The quadrature grows
#: with the budget so that the integration stays ahead of the approximation;
#: the appendix records that raising it further does not move the result.
DEFAULT_RFM = ((100, 14), (200, 16), (400, 18), (600, 20), (900, 22))

#: Mesh refinements of the finite element baseline.
DEFAULT_FEM = (1, 2, 3, 4)

#: (Bernstein degree, collocation points per axis) for Eig-PIELM.
DEFAULT_PIELM = ((6, 35), (8, 45), (9, 50), (10, 45))

#: The probe budget and quadrature level the blind scale rule runs at.
SCALE_PROBE = (400, 18)


def select_scale_blind(
    domain,
    features: int,
    quadrature_level: int,
    scales,
    count: int,
    probe_seeds: tuple[int, int] = (101, 102),
) -> tuple[float, float]:
    """Choose the frequency scale without looking at any exact value.

    Two independent draws of the same size should agree if the space resolves
    the modes and disagree if it does not, so the scale whose Ritz values move
    least between draws is the one selected.  The exact spectrum is never
    consulted, which is what keeps the tuning from being a second fit to the
    answer.
    """
    best_scale, best_discrepancy = None, np.inf
    for scale in scales:
        try:
            runs = [
                rfm_solve(
                    domain,
                    features=features,
                    seed=seed,
                    scale=scale,
                    quadrature_level=quadrature_level,
                    count=count,
                ).values
                for seed in probe_seeds
            ]
        except (np.linalg.LinAlgError, RuntimeError) as error:
            warnings.warn(f"scale {scale:g} could not be solved: {error}", RuntimeWarning, stacklevel=2)
            continue
        discrepancy = float(np.max(np.abs(runs[0] - runs[1]) / np.abs(runs[0])))
        if discrepancy < best_discrepancy:
            best_scale, best_discrepancy = scale, discrepancy
    if best_scale is None:
        raise RuntimeError("no candidate scale produced a solve")
    return best_scale, best_discrepancy


def _timed_median(call, repeats: int):
    """Repeat a call and keep the median clock; the values do not move."""
    seconds, result = [], None
    for _ in range(repeats):
        result = call()
        seconds.append(result.seconds)
    return result, float(np.median(seconds))


def rfm_row(domain, features, scale, quadrature_level, seeds, count, repeats):
    exact = domain.exact_ordered(count)
    errors, seconds, retained, per_draw = [], [], [], []
    starts = domain.multiple_level_starts(count)
    for seed in seeds:
        result, elapsed = _timed_median(
            functools.partial(
                rfm_solve,
                domain,
                features=features,
                seed=seed,
                scale=scale,
                quadrature_level=quadrature_level,
                count=count,
            ),
            repeats,
        )
        errors.append(float(np.max(np.abs(result.values - exact) / exact)))
        seconds.append(elapsed)
        retained.append(result.retained)
        per_draw.append(
            [_cluster(domain, result.values, start, count) for start in starts]
        )

    grouped = list(zip(*per_draw, strict=True)) if per_draw and per_draw[0] else []
    clusters = [
        {
            "level": group[0]["level"],
            "multiplicity": group[0]["multiplicity"],
            "identified": int(sum(item["identified"] for item in group)),
            "relative_width": float(np.median([i["relative_width"] for i in group])),
            "separation": float(np.median([i["separation"] for i in group])),
        }
        for group in grouped
    ]
    return {
        "method": "RFM",
        "setting": f"N={features}",
        "features": features,
        "scale": scale,
        "quadrature_level": quadrature_level,
        "draws": len(seeds),
        "median_error": float(np.median(errors)),
        "min_error": float(np.min(errors)),
        "max_error": float(np.max(errors)),
        "median_seconds": float(np.median(seconds)),
        "min_seconds": float(np.min(seconds)),
        "max_seconds": float(np.max(seconds)),
        "median_retained": float(np.median(retained)),
        "clusters": clusters,
    }


def _cluster(domain, values: np.ndarray, index: int, count: int) -> dict[str, Any]:
    """Score the ordered Ritz values against the exact level at ``index``.

    ``separation`` is the width of the computed cluster divided by its distance
    to the nearest neighbouring level.  Well below one means the cluster is
    resolved as a cluster; of order one means it has been smeared into its
    neighbours, which is the failure Lemma 3.3 excludes by a threshold.
    """
    exact = domain.exact_ordered(count)
    start, stop = domain.cluster_of(index, count)
    block = np.asarray(values[start:stop])
    level = float(exact[start])
    width = float(block.max() - block.min())
    below = float(exact[start - 1]) if start > 0 else None
    above = float(exact[stop]) if stop < len(exact) else None
    gap = min(
        level - below if below is not None else np.inf,
        above - level if above is not None else np.inf,
    )
    identified = bool(
        (below is None or block.min() > 0.5 * (below + level))
        and (above is None or block.max() < 0.5 * (level + above))
    )
    return {
        "level": level,
        "multiplicity": stop - start,
        "relative_width": width / level,
        "separation": width / gap if np.isfinite(gap) and gap > 0 else np.inf,
        "identified": identified,
    }


def fem_row(domain, subdivisions, count, repeats):
    exact = domain.exact_ordered(count)
    result, elapsed = _timed_median(
        lambda: isoparametric_fem.solve(domain, subdivisions, count=count), repeats
    )
    return {
        "method": "P2 FEM",
        "setting": f"n={subdivisions}",
        "subdivisions": subdivisions,
        "degrees_of_freedom": result.dimension,
        "median_error": float(np.max(np.abs(result.values - exact) / exact)),
        "median_seconds": elapsed,
        # kept so that a slow baseline can be told from a badly solved one
        "assemble_seconds": result.assemble_seconds,
        "solve_seconds": result.solve_seconds,
    }


def pielm_row(domain, degree, grid, count, repeats):
    """Eig-PIELM at one polynomial degree.

    The basis is deterministic, so there are no draws to average over; only the
    clock is repeated.  The dimension that matters for the comparison is the
    admissible one, which is smaller than the number of basis functions.
    """
    exact = domain.exact_ordered(count)
    try:
        result, elapsed = _timed_median(
            lambda: eig_pielm.solve(
                domain, degree=degree, count=count, grid_per_axis=grid
            ),
            repeats,
        )
    except (np.linalg.LinAlgError, RuntimeError) as error:
        raise RuntimeError(f"Eig-PIELM deg={degree}, grid={grid} failed") from error
    error = (
        float("inf")
        if len(result.values) < count
        else float(np.max(np.abs(result.values - exact) / exact))
    )
    return {
        "method": "Eig-PIELM",
        "setting": f"deg={degree}",
        "degree": degree,
        "grid_per_axis": grid,
        "basis_functions": result.dimension,
        "admissible": result.admissible,
        "median_error": error,
        "min_error": error,
        "max_error": error,
        "median_seconds": elapsed,
    }


def verdict(rows) -> dict[str, Any]:
    """Whether the sampled space is both faster and more accurate than each
    baseline, and by how much.

    ``speed_factor_at_matched_accuracy`` is the honest speed number: the time
    of the baseline's best setting divided by the time of the cheapest sampled
    budget that is already at least as accurate.  Comparing the two best
    settings alone would flatter whichever method happened to be given a larger
    budget.
    """
    ours = [row for row in rows if row["method"] == "RFM"]
    best_ours = min(ours, key=lambda row: row["median_error"])
    summary: dict[str, Any] = {}
    for name in ("P2 FEM", "Eig-PIELM"):
        group = [
            row
            for row in rows
            if row["method"] == name and np.isfinite(row["median_error"])
        ]
        if not group:
            summary[name] = {"beaten": False, "reason": "no converged baseline point"}
            continue
        best = min(group, key=lambda row: row["median_error"])
        cheaper = [r for r in ours if r["median_error"] <= best["median_error"]]
        matched = min(cheaper, key=lambda r: r["median_seconds"]) if cheaper else None
        summary[name] = {
            "baseline": best["setting"],
            "baseline_error": best["median_error"],
            "baseline_seconds": best["median_seconds"],
            "our_best_error": best_ours["median_error"],
            "our_best_seconds": best_ours["median_seconds"],
            "accuracy_factor": best["median_error"] / best_ours["median_error"],
            "speed_factor_at_matched_accuracy": (
                best["median_seconds"] / matched["median_seconds"] if matched else None
            ),
            "matched_setting": matched["setting"] if matched else None,
            "beaten": bool(
                matched is not None
                and best_ours["median_error"] < best["median_error"]
                and matched["median_seconds"] < best["median_seconds"]
            ),
        }
    return summary


def quadrature_sensitivity(domain, features, scale, levels, count, seed=1):
    """Is the reported accuracy the trial space's, or the quadrature's?

    The same draw is integrated on three rules; if the answer does not move,
    the error being reported belongs to the approximation and not to the sum
    that stands in for the integral.
    """
    exact = domain.exact_ordered(count)
    records = []
    for level in levels:
        values = rfm_solve(
            domain,
            features=features,
            seed=seed,
            scale=scale,
            quadrature_level=level,
            count=count,
        ).values
        records.append(
            {
                "level": level,
                "error": float(np.max(np.abs(values - exact) / exact)),
            }
        )
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--contrast", type=float, default=GradedBall.DEFAULT_CONTRAST)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args(argv)

    provenance.require_threads(args.threads)
    domain = GradedBall(contrast=args.contrast)
    count = args.count
    seeds = list(range(1, args.draws + 1))

    config = {
        "contrast": args.contrast,
        "count": count,
        "draws": args.draws,
        "repeats": args.repeats,
        "scales": list(DEFAULT_SCALES),
        "scale_probe": list(SCALE_PROBE),
        "fem": list(DEFAULT_FEM),
        "pielm": [list(item) for item in DEFAULT_PIELM],
        "rfm": [list(item) for item in DEFAULT_RFM],
        "threads": args.threads,
    }
    run = provenance.open_run("experiment2", config=config, run_id=args.run_id, resume=args.resume)
    started = time.perf_counter()

    exact = domain.exact_ordered(count)
    print("exact: " + " ".join(f"{value:.8f}" for value in exact), flush=True)

    tuning = run.completed("scale_selection")
    if tuning is None:
        scale, discrepancy = select_scale_blind(
            domain, SCALE_PROBE[0], SCALE_PROBE[1], DEFAULT_SCALES, count
        )
        run.complete("scale_selection", {"scale": scale, "discrepancy": discrepancy})
    else:
        scale, discrepancy = tuning["scale"], tuning["discrepancy"]
    print(
        f"blind scale selection: {scale:g} "
        f"(draw-to-draw {discrepancy:.2e}); the baselines are tuned per setting",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for subdivisions in DEFAULT_FEM:
        call_id = f"fem_n{subdivisions}"
        row = run.completed(call_id)
        if row is None:
            row = run.complete(call_id, fem_row(domain, subdivisions, count, args.repeats))
        rows.append(row)
        print(
            f"  P2 FEM     {row['setting']:9s} dof={row['degrees_of_freedom']:7d} "
            f"t={row['median_seconds']:8.3f}s err={row['median_error']:.3e}",
            flush=True,
        )
    for degree, grid in DEFAULT_PIELM:
        call_id = f"pielm_degree{degree}_grid{grid}"
        row = run.completed(call_id)
        if row is None:
            row = run.complete(call_id, pielm_row(domain, degree, grid, count, args.repeats))
        rows.append(row)
        print(
            f"  Eig-PIELM  {row['setting']:9s} adm={row['admissible']:7d} "
            f"t={row['median_seconds']:8.3f}s err={row['median_error']:.3e}",
            flush=True,
        )
    for features, level in DEFAULT_RFM:
        call_id = f"rfm_N{features}_q{level}"
        row = run.completed(call_id)
        if row is None:
            row = run.complete(call_id, rfm_row(domain, features, scale, level, seeds, count, args.repeats))
        rows.append(row)
        print(
            f"  RFM        {row['setting']:9s} ret={row['median_retained']:7.0f} "
            f"t={row['median_seconds']:8.3f}s err={row['median_error']:.3e} "
            f"[{row['min_error']:.1e},{row['max_error']:.1e}]  "
            + "  ".join(
                f"[x{c['multiplicity']} {c['identified']}/{row['draws']}]"
                for c in row["clusters"]
            ),
            flush=True,
        )

    scored = verdict(rows)
    largest = DEFAULT_RFM[-1]
    sensitivity_record = run.completed("quadrature_sensitivity")
    if sensitivity_record is None:
        sensitivity_record = run.complete("quadrature_sensitivity", {"rows": quadrature_sensitivity(
            domain, largest[0], scale, (largest[1], largest[1] + 6, largest[1] + 14), count
        )})
    sensitivity = sensitivity_record["rows"]
    passed = all(item.get("beaten") for item in scored.values())

    provenance.write_json(
        run.path("result.json"),
        {
            "problem": domain.name,
            "contrast": domain.contrast,
            "exact": exact.tolist(),
            "tuning": {
                "rfm_scale": scale,
                "rfm_probe_discrepancy": discrepancy,
                "rfm_rule": "blind: least draw-to-draw movement, "
                "exact spectrum not used",
                "baseline_rule": "each baseline at the setting minimizing its "
                "own error",
            },
            "rows": rows,
            "verdict": scored,
            "quadrature_sensitivity": sensitivity,
            "passed": passed,
        },
        overwrite=True,
    )
    provenance.write_csv(run.path("rows.csv"), [
        {k: v for k, v in row.items() if not isinstance(v, list)} for row in rows
    ])
    run.seal(rows=len(rows), comparison_passed=passed)
    print(
        f"verdict: {'PASS' if passed else 'FAIL'}  "
        f"[{time.perf_counter() - started:.0f}s]  -> {run.directory}",
        flush=True,
    )


if __name__ == "__main__":
    main()
