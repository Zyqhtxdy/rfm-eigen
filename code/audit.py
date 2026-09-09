"""Audit recorded statistics, captions, hashes, and independently rebuilt tables.

Table builders write into a temporary directory. The audit never repairs an
input or overwrites a table; only its JSON report is written to the output path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from paper_assets.paths import DATA as OUTPUT_DATA
from paper_assets.paths import REPO_DATA as DATA
from paper_assets.paths import ROOT
from paper_assets.records import draw_counts, read_converged, repetitions_per_draw

WORDS = {"three": 3, "five": 5, "ten": 10, "twenty": 20}
TABLES = [f"experiment{i}_rewrite_summary_table.tex" for i in (1, 2, 4)] + [
    "experiment5_nonlinear_gpe_table.tex", "experiment6_dipolar_bec_table.tex"
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def caption_of(name: str) -> str:
    text = (DATA / name).read_text(encoding="utf-8")
    start = text.index(r"\caption{") + len(r"\caption{")
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return " ".join(text[start:index].split())
    raise ValueError(f"unclosed caption in {name}")


def rfm_draw_count_in(caption: str) -> int | None:
    match = re.search(r"\b(\d+|three|five|ten|twenty) independent feature draws\b", caption)
    if match is None:
        return None
    word = match.group(1)
    return int(word) if word.isdecimal() else WORDS[word]


def variable_draw_counts_in(caption: str) -> dict[int, int]:
    match = re.search(r"sample sizes for.*?N=([\d, ]+).*?are ([\d, ]+), respectively", caption)
    if match is None:
        raise ValueError("caption does not state the per-budget sample counts")
    budgets = [int(v) for v in match.group(1).split(",")]
    counts = [int(v) for v in match.group(2).split(",")]
    return dict(zip(budgets, counts, strict=True))


def _uniform_count(counts) -> int:
    values = set(int(v) for v in counts)
    if len(values) != 1:
        raise ValueError(f"draw counts differ between configurations: {values}")
    return values.pop()


def check_draw_counts() -> dict:
    exp1 = pd.read_csv(DATA / "experiment1_theory_matched_trials.csv")
    exp2 = json.loads((DATA / "experiment2_graded_ball_run.json").read_text())
    exp4 = pd.read_csv(DATA / "runs/experiment4/rfm_rows.csv")
    exp5 = read_converged(DATA / "experiment5_rfm_timed.csv")
    exp6 = read_converged(DATA / "experiment6_rfm_timed.csv")
    counts = [_uniform_count(exp1.groupby("N").size()),
              _uniform_count(row["draws"] for row in exp2["rows"] if row["method"] == "RFM"),
              int(exp4.seed.nunique()), draw_counts(exp5), _uniform_count(draw_counts(exp6).values())]
    report = {}
    for name, actual in zip(TABLES, counts, strict=True):
        caption = caption_of(name)
        stated = variable_draw_counts_in(caption) if isinstance(actual, dict) else rfm_draw_count_in(caption)
        if actual != stated:
            raise ValueError(f"{name}: caption {stated}, recorded {actual}")
        report[name] = {"claimed_in_caption": stated, "recorded_in_run": actual, "match": True}
    repeats = repetitions_per_draw(exp5)
    repeat_match = re.search(r"with (\d+|three|five|ten|twenty) timed repetitions per draw", caption_of(TABLES[3]))
    if repeat_match is None:
        raise ValueError("Example 5 caption omits its timing repetitions")
    word = repeat_match.group(1)
    stated_repeats = int(word) if word.isdecimal() else WORDS[word]
    if repeats != stated_repeats:
        raise ValueError(f"Example 5 repeats: caption {stated_repeats}, recorded {repeats}")
    baseline = read_converged(DATA / "experiment6_gflm_ktm_timed.csv")
    baseline_count = _uniform_count(baseline.groupby("mesh_width").pair_index.nunique())
    if baseline_count != 20 or "twenty initial states" not in caption_of(TABLES[4]):
        raise ValueError("Example 6 initial-state counts differ from its caption")
    return report


def check_experiment1() -> dict:
    from rfmeig.experiments.exp1_interior_double import (
        DEFAULT_FEATURE_COUNTS,
        DEFAULT_REFERENCE_VALUE,
    )

    fields = ("h1_gap", "eta_f_h1", "cluster_relative_error")
    frames = {}
    for role, prefix, draws in (("validation", "experiment1_theory_matched", 100),
                               ("calibration", "experiment1_rate_calibration", 20)):
        frame = pd.read_csv(DATA / f"{prefix}_trials.csv", float_precision="round_trip")
        summary = pd.read_csv(DATA / f"{prefix}_summary.csv", float_precision="round_trip").set_index("N")
        metadata = json.loads((DATA / f"{prefix}_metadata.json").read_text())
        if frame.duplicated(["N", "sample"]).any() or frame.seed.duplicated().any():
            raise ValueError(f"{role}: duplicate draws")
        if set(frame.N) != set(DEFAULT_FEATURE_COUNTS) or not frame.groupby("N").size().eq(draws).all():
            raise ValueError(f"{role}: inconsistent draw count")
        if not frame.order_per_subinterval.eq(512).all():
            raise ValueError(f"{role}: data are not uniformly at order 512")
        if any(metadata[f"{stage}_composite_order_per_interval"] != 512
               for stage in ("assembly", "evaluation", "subspace")):
            raise ValueError(f"{role}: stale quadrature metadata")
        if metadata['lambda_star'] != DEFAULT_REFERENCE_VALUE or metadata['reported_statistic'] != 'arithmetic mean':
            raise ValueError(f"{role}: stale reference or reported statistic")
        if set(summary.index) != set(frame.N):
            raise ValueError(f"{role}: summary budgets differ from trials")
        for field in fields:
            grouped = frame.groupby("N")[field]
            for statistic, values in (("mean", grouped.mean()), ("median", grouped.median()),
                                      ("q90", grouped.quantile(.9))):
                np.testing.assert_allclose(summary.loc[values.index, f"{field}_{statistic}"], values,
                                           rtol=1e-12, atol=0, err_msg=f"{role}: stale {field}_{statistic}")
                if statistic in ('mean', 'q90'):
                    slope = np.polyfit(np.log(values.index), np.log(values.to_numpy()), 1)[0]
                    np.testing.assert_allclose(metadata[f'{field}_{statistic}_slope'], slope, rtol=1e-12)
        frames[role] = frame
    if set(frames["validation"].seed) & set(frames["calibration"].seed):
        raise ValueError("calibration and validation share seeds")
    guides = json.loads((DATA / "experiment1_rate_guides.json").read_text())
    for field, power, key in (("h1_gap", .5, "gap_constant"),
                               ("cluster_relative_error", 1., "eigenvalue_constant")):
        means = frames["calibration"].groupby("N")[field].mean()
        constant = guides["safety_factor"] * np.max(means.index.to_numpy() ** power * means.to_numpy())
        np.testing.assert_allclose(guides[key], constant, rtol=1e-12, atol=0)
    return {"validation_draws": len(frames["validation"]),
            "calibration_draws": len(frames["calibration"]), "order": 512,
            "summaries_recomputed": True, "calibration_recomputed": True}


def check_experiment1_quadrature() -> dict:
    fields = ['h1_gap', 'eta_f_h1', 'cluster_relative_error']
    prefix = DATA / 'experiment1_theory_matched_quadrature_check'
    trials = pd.read_csv(str(prefix) + '_trials.csv', float_precision='round_trip')
    summary = pd.read_csv(str(prefix) + '_summary.csv', float_precision='round_trip')
    keys = ['group', 'order_per_subinterval', 'N']
    if len(trials) != 2880 or trials.duplicated(keys + ['sample']).any():
        raise ValueError('quadrature check has missing or duplicate solves')
    for role, name, draws in [('validation', 'theory_matched', 100), ('calibration', 'rate_calibration', 20)]:
        group = trials[trials.group == role]
        if set(group.order_per_subinterval) != {128, 256, 512}:
            raise ValueError(f'{role}: unexpected quadrature orders')
        if len(group.groupby(['N', 'order_per_subinterval'])) != 24 or not group.groupby(['N', 'order_per_subinterval']).size().eq(draws).all():
            raise ValueError(f'{role}: incomplete quadrature refinement')
        recorded = pd.read_csv(DATA / f'experiment1_{name}_trials.csv', float_precision='round_trip').set_index('seed').sort_index()
        fine = group[group.order_per_subinterval == 512].set_index('seed').sort_index()
        if not recorded.index.equals(fine.index):
            raise ValueError(f'{role}: refined seeds differ from the reported run')
        np.testing.assert_allclose(recorded[fields], fine[fields], rtol=1e-12, atol=0)
    stored = summary.set_index(keys).sort_index()
    for field in fields:
        grouped = trials.groupby(keys)[field]
        for statistic, values in [('mean', grouped.mean()), ('median', grouped.median()), ('q90', grouped.quantile(.9))]:
            np.testing.assert_allclose(stored.loc[values.index, f'{field}_{statistic}'], values, rtol=1e-12, atol=0)
    return {'solves': len(trials), 'orders': [128, 256, 512], 'summary_rows': len(stored), 'matched': True}


def check_experiment3() -> dict:
    from rfmeig.experiments.exp3_high_dimension import summarize_rfm

    rows = pd.read_csv(DATA / "runs/experiment3_independent/rfm_trials.csv", float_precision="round_trip")
    stored = pd.read_csv(DATA / "runs/experiment3_independent/rfm_summary.csv", float_precision="round_trip")
    if rows.duplicated(['potential','N','trial']).any() or len(rows)!=160:
        raise ValueError('Example 3: incomplete or repeated feature draws')
    if not rows.groupby(['potential','N']).size().eq(20).all():
        raise ValueError('Example 3: each budget requires twenty draws')
    baseline=pd.read_csv(DATA/'experiment3_drm_independent.csv')
    if len(baseline)!=6 or baseline.duplicated(['potential','mode']).any():
        raise ValueError('Example 3: expected six fixed baseline networks')
    np.testing.assert_allclose(baseline.lambda_final,
        (baseline.domain_energy+2000*baseline.boundary_mean)/baseline.mass,rtol=1e-12)
    np.testing.assert_allclose(baseline.relative_error,
        abs(baseline.lambda_final-baseline.lambda_ref)/baseline.lambda_ref,rtol=1e-12)
    count = int(stored.eigen_index.max())
    rebuilt = pd.DataFrame(summarize_rfm(rows.to_dict("records"), count))
    keys = ["potential", "N", "eigen_index"]
    stored, rebuilt = (frame.set_index(keys).sort_index() for frame in (stored, rebuilt))
    np.testing.assert_allclose(stored[rebuilt.columns], rebuilt, rtol=1e-12, atol=0)
    return {"trials": len(rows), "summary_rows": len(rebuilt), "matched": True}


def rebuild_tables(output: Path) -> dict:
    from iaea_repro.paper_assets import build_experiment4_table
    from paper_assets import assets

    builders = (assets.build_experiment1_table, assets.build_experiment2_table,
                build_experiment4_table, assets.build_experiment5_table, assets.build_experiment6_table)
    report = {}
    for name, builder in zip(TABLES, builders, strict=True):
        builder(output_directory=output)
        before, after = sha256(DATA / name), sha256(output / name)
        report[name] = {"sha256": before, "rebuilt": True, "unchanged_by_rebuild": before == after}
        if before != after:
            raise ValueError(f"{name}: rebuilding from recorded inputs changes the table")
    return report


def check_hashes() -> dict:
    frame = pd.read_csv(DATA / "core_artifact_sha256.csv")
    failures = []
    for row in frame.itertuples():
        path = (ROOT / row.path).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or sha256(path) != row.sha256:
            failures.append(row.path)
    if failures:
        raise ValueError("core artifact hash mismatch: " + ", ".join(failures))
    exports = {}
    for manifest in sorted((DATA / "runs").glob("*/output_manifest.json")):
        record = json.loads(manifest.read_text())
        omitted = []
        for name, expected in record["files"].items():
            path = manifest.parent / name
            if not path.exists() and name.startswith("calls/"):
                omitted.append(name)
            elif not path.is_file() or sha256(path) != expected:
                raise ValueError(f"archived run differs from its original manifest: {path}")
        exports[manifest.parent.name] = {"packaged_files_verified": len(record["files"]) - len(omitted),
                                        "per_call_logs_not_packaged": len(omitted)}
    return {"core_files_verified": len(frame), "archived_exports": exports}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT_DATA / "reproducibility_audit.json")
    args = parser.parse_args(argv)
    report, failures = {}, []
    with tempfile.TemporaryDirectory(prefix="rfmeig-audit-") as temporary:
        checks = {"draw_counts": check_draw_counts, "experiment1": check_experiment1,
                  "experiment1_quadrature": check_experiment1_quadrature,
                  "experiment3": check_experiment3, "hashes": check_hashes,
                  "tables": lambda: rebuild_tables(Path(temporary))}
        for name, check in checks.items():
            try:
                report[name] = check()
                print(f"  ok    {name}")
            except (AssertionError, ValueError, OSError, KeyError) as error:
                failures.append(f"{name}: {error}")
                report[name] = {"error": str(error)}
                print(f"  FAIL  {name}: {error}")
    report.update(passed=not failures, failures=failures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
