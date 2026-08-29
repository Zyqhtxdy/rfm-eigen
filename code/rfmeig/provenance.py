"""Run directories, hashing and atomic writes.

Every experiment records enough to say where a number came from: the hash of the
code that produced it, the configuration, the seeds, and the wall times. The six
experiments previously each carried their own copy of this machinery; it is
collected here so that a run of any of them is laid out the same way.

A run is a directory under ``rfm_eigen_data/<experiment>/<run id>``. It is
*sealed* once ``seal`` has been called, after which it is never modified: a
second attempt to write into it raises rather than overwriting recorded numbers.
An interrupted run can be resumed, because each call writes its own completion
marker and a completed call is loaded rather than recomputed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
#: A rerun writes here.  The recorded output the paper was written
#: from is in ``code/data``; this is kept separate so that a rerun
#: cannot quietly overwrite it.
DATA_ROOT = PACKAGE_ROOT.parents[1] / "output"

_SEAL = "output_manifest.json"


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------
def canonical_json(value: Any) -> str:
    """Serialize so that two equal configurations hash equally."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_source() -> str:
    """Hash every module of the package, so a run names the code that made it."""
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(PACKAGE_ROOT).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


# --------------------------------------------------------------------------
# atomic writes
# --------------------------------------------------------------------------
def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(path.suffix + ".partial")
    scratch.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(scratch, path)


def write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(".npz.partial")
    with scratch.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(scratch, path)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write a table whose columns are the union of the keys, in first-seen order."""
    import csv

    rows = list(rows)
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(path.suffix + ".partial")
    with scratch.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
    os.replace(scratch, path)


# --------------------------------------------------------------------------
# runs
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Run:
    """One execution of one experiment, on disk."""

    experiment: str
    directory: Path
    code_sha256: str
    started: float

    @property
    def sealed(self) -> bool:
        return (self.directory / _SEAL).exists()

    def path(self, *parts: str) -> Path:
        target = self.directory.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    # -- per-call completion markers, so an interrupted run can be resumed ---
    def _marker(self, call_id: str) -> Path:
        return self.directory / "calls" / f"{call_id}.json"

    def completed(self, call_id: str) -> dict[str, Any] | None:
        marker = self._marker(call_id)
        if not marker.exists():
            return None
        return json.loads(marker.read_text(encoding="utf-8"))

    def complete(self, call_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
        record = dict(row)
        record.setdefault("recorded_at", time.time())
        write_json(self._marker(call_id), record, overwrite=True)
        return record

    def seal(self, **summary: Any) -> None:
        """Close the run and hash everything it wrote."""
        if self.sealed:
            raise RuntimeError(f"{self.directory} is already sealed")
        files = {
            path.relative_to(self.directory).as_posix(): hash_file(path)
            for path in sorted(self.directory.rglob("*"))
            if path.is_file()
        }
        write_json(
            self.directory / _SEAL,
            {
                "experiment": self.experiment,
                "code_sha256": self.code_sha256,
                "wall_seconds": time.perf_counter() - self.started,
                "files": files,
                **summary,
            },
        )


def open_run(
    experiment: str,
    *,
    config: Mapping[str, Any],
    run_id: str | None = None,
    resume: bool = False,
) -> Run:
    """Create, or reopen, the directory that a run writes into."""
    root = DATA_ROOT / experiment
    if run_id is None:
        run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    directory = root / run_id
    code = hash_source()

    if directory.exists() and any(directory.iterdir()):
        if not resume:
            raise FileExistsError(
                f"run {run_id} already exists; pass resume=True to continue it"
            )
        if (directory / _SEAL).exists():
            raise RuntimeError(f"run {run_id} is sealed and cannot be resumed")
        recorded = json.loads(
            (directory / "provenance.json").read_text(encoding="utf-8")
        )
        if recorded["code_sha256"] != code:
            raise RuntimeError(
                "resume refused: the code has changed since the run began"
            )
        if canonical_json(recorded["config"]) != canonical_json(dict(config)):
            raise RuntimeError("resume refused: the configuration has changed")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        write_json(
            directory / "provenance.json",
            {
                "experiment": experiment,
                "run_id": run_id,
                "code_sha256": code,
                "config": dict(config),
                "config_sha256": hash_bytes(canonical_json(dict(config)).encode()),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "host": socket.gethostname(),
                "threads": {
                    name: os.environ.get(name, "unset")
                    for name in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                    )
                },
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
    return Run(experiment, directory, code, time.perf_counter())


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------
@dataclass
class Timer:
    """Wall time of one call, excluding everything written afterwards."""

    seconds: float = 0.0

    def __enter__(self) -> Timer:
        self._started = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.seconds = time.perf_counter() - self._started


def timed(function, /, *args, **kwargs) -> tuple[Any, float]:
    """Return the value and the wall time, so callers cannot forget the clock."""
    started = time.perf_counter()
    value = function(*args, **kwargs)
    return value, time.perf_counter() - started


def require_threads(count: int) -> int:
    """Check that the thread count in force is the one that was asked for.

    Pinning happens in :mod:`rfmeig`, before anything imports NumPy, because the
    threading libraries read their variables when they load and ignore later
    changes.  This cannot therefore set the count; it can only refuse to run at
    the wrong one, which is what matters -- a timing comparison recorded under a
    thread count it did not run at is worse than no timing at all.

    Set it with ``RFMEIG_THREADS`` in the environment.
    """
    from rfmeig import PINNED_THREADS

    if int(count) != int(PINNED_THREADS):
        raise RuntimeError(
            f"--threads {count} was requested but {PINNED_THREADS} thread(s) "
            f"are in force; the threading libraries fix this when they load, so "
            f"set it in the environment before starting, for example "
            f"RFMEIG_THREADS={count}"
        )
    return int(PINNED_THREADS)


def quantile_summary(
    values: Iterable[float], levels: Iterable[float]
) -> dict[str, float]:
    """Median, extremes and the requested upper quantiles of repeated draws."""
    sample = np.asarray(sorted(float(v) for v in values), dtype=float)
    if sample.size == 0:
        raise ValueError("no samples")
    summary = {
        "count": int(sample.size),
        "median": float(np.median(sample)),
        "min": float(sample[0]),
        "max": float(sample[-1]),
    }
    for level in levels:
        summary[f"q{int(round(100 * level)):02d}"] = float(np.quantile(sample, level))
    return summary


def iter_seeds(base: int, count: int) -> Iterator[int]:
    """Consecutive seeds from a stated base, so a run's draws are reproducible."""
    for offset in range(count):
        yield base + offset
