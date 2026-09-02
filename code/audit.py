"""Check that the committed tables still follow from the committed runs.

Two things are verified, and both of them are things a reader can check for
themselves.

First, every table whose builder runs from the committed data is rebuilt and
compared, byte for byte, with the copy in ``data``.  A table that no longer
follows from its run fails here rather than in review.  Two tables are recorded
by hash alone: Table 5, whose builder reads the reactor checkpoints under
``artifacts``, and Table 1, whose builder also writes a figure and would leave
the working tree modified on every run.

Second, the number of draws each caption claims is compared with the number of
draws the run actually holds.  This is the check that matters most: a caption
saying twenty over a run of five is not visible in any hash, and it is exactly
the kind of drift that accumulates when an experiment is rerun at a different
budget and only some of the files are refreshed.

Run it from this directory::

    python audit.py

It writes ``data/reproducibility_audit.json`` and exits non-zero on failure.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
RUNS = DATA / "runs"

#: caption word -> count, for the counts the captions actually use
WORDS = {"five": 5, "ten": 10, "twenty": 20, "100": 100}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# how many draws each run holds
# ---------------------------------------------------------------------------


def _draws_experiment1() -> int:
    frame = pd.read_csv(DATA / "experiment1_theory_matched_trials.csv")
    counts = set(frame.groupby("N").size())
    if len(counts) != 1:
        raise AssertionError(f"Example 1 budgets differ in draw count: {counts}")
    return counts.pop()


def _draws_experiment2() -> int:
    record = json.loads((DATA / "experiment2_graded_ball_run.json").read_text("utf-8"))
    counts = {int(row["draws"]) for row in record["rows"] if "draws" in row}
    if len(counts) != 1:
        raise AssertionError(f"Example 2 rows differ in draw count: {counts}")
    return counts.pop()


def _draws_experiment3() -> int:
    frame = pd.read_csv(RUNS / "experiment3" / "rfm_summary.csv")
    counts = set(frame["trials"])
    if len(counts) != 1:
        raise AssertionError(f"Example 3 rows differ in draw count: {counts}")
    return int(counts.pop())


def _draws_experiment4() -> int:
    return int(pd.read_csv(RUNS / "experiment4" / "rfm_rows.csv")["seed"].nunique())


def _draws_experiment5() -> int:
    return int(pd.read_csv(DATA / "experiment5_rfm_timed.csv")["seed"].nunique())


def _draws_experiment6() -> int:
    return int(pd.read_csv(DATA / "experiment6_rfm_timed.csv")["seed"].nunique())


#: table -> (the run's draw count, the count its caption must state)
CLAIMS = [
    ("experiment1_rewrite_summary_table.tex", _draws_experiment1,
     "data/experiment1_theory_matched_trials.csv"),
    ("experiment2_rewrite_summary_table.tex", _draws_experiment2,
     "data/experiment2_graded_ball_run.json"),
    ("experiment3_rewrite_summary_table.tex", _draws_experiment3,
     "data/runs/experiment3/rfm_summary.csv"),
    ("experiment4_rewrite_summary_table.tex", _draws_experiment4,
     "data/runs/experiment4/rfm_rows.csv"),
    ("experiment5_nonlinear_gpe_table.tex", _draws_experiment5,
     "data/experiment5_rfm_timed.csv"),
    ("experiment6_dipolar_bec_table.tex", _draws_experiment6,
     "data/experiment6_rfm_timed.csv"),
]


def caption_of(name: str) -> str:
    text = (DATA / name).read_text(encoding="utf-8")
    start = text.index("\\caption{")
    return " ".join(text[start:text.index("\n", start)].split())


def rfm_draw_count_in(caption: str) -> int | None:
    """The count the caption gives for the random feature draws.

    Captions that also count something else -- the initial states of a
    baseline, for instance -- state the feature draws first, so the first
    number word before "draws" is the one to read.
    """
    head = caption.split("draws")[0]
    found = [(head.rfind(word), value) for word, value in WORDS.items()
             if word in head]
    if not found:
        return None
    return max(found)[1]


def _builders() -> dict:
    """The table builders that run from the committed data alone."""
    from paper_assets import assets

    assets.use_plot_style()
    return {
        "experiment2_rewrite_summary_table.tex": assets.build_experiment2_table,
        "experiment3_rewrite_summary_table.tex": assets.build_experiment3_table,
        "experiment5_nonlinear_gpe_table.tex": assets.build_experiment5_table,
        "experiment6_dipolar_bec_table.tex": assets.build_experiment6_table,
    }


BUILDERS = _builders()


# ---------------------------------------------------------------------------


def main() -> int:
    report: dict[str, object] = {
        "draw_counts": {},
        "tables": {},
    }
    failures: list[str] = []

    for name, counter, source in CLAIMS:
        recorded = counter()
        claimed = rfm_draw_count_in(caption_of(name))
        ok = claimed == recorded
        report["draw_counts"][name] = {
            "claimed_in_caption": claimed,
            "recorded_in_run": recorded,
            "source": source,
            "match": ok,
        }
        before = sha256(DATA / name)
        builder = BUILDERS.get(name)
        if builder is None:
            report["tables"][name] = {"sha256": before, "rebuilt": False}
        else:
            builder()
            after = sha256(DATA / name)
            report["tables"][name] = {"sha256": after, "rebuilt": True,
                                      "unchanged_by_rebuild": before == after}
            if before != after:
                failures.append(f"{name}: rebuilding it changed the file")
        if not ok:
            failures.append(
                f"{name}: caption claims {claimed} draws, "
                f"{source} records {recorded}"
            )

    report["passed"] = not failures
    (DATA / "reproducibility_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    for name, entry in report["draw_counts"].items():
        mark = "ok  " if entry["match"] else "FAIL"
        print(f"  {mark}  {name:44s} "
              f"caption {entry['claimed_in_caption']}, "
              f"run {entry['recorded_in_run']}")
    if failures:
        print("\n".join("  " + f for f in failures))
    print(f"\n  {'passed' if report['passed'] else 'FAILED'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
