"""Read convergence flags and draw counts without conflating draws with repeats."""

from pathlib import Path

import pandas as pd


def converged_rows(frame: pd.DataFrame) -> pd.DataFrame:
    flags = frame["converged"].astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}
    parsed = flags.map(mapping)
    if parsed.isna().any():
        raise ValueError("missing or unrecognized convergence flag")
    return frame.loc[parsed.astype(bool)]


def read_converged(path: Path) -> pd.DataFrame:
    return converged_rows(pd.read_csv(path, float_precision="round_trip"))


def draw_counts(frame: pd.DataFrame, budget: str = "features") -> dict[int, int]:
    return {int(n): int(count) for n, count in frame.groupby(budget)["seed"].nunique().items()}


def repetitions_per_draw(frame: pd.DataFrame, budget: str = "features") -> int:
    counts = frame.groupby([budget, "seed"]).size()
    if counts.empty or counts.nunique() != 1:
        raise ValueError("timed repetitions differ between draws")
    return int(counts.iloc[0])
