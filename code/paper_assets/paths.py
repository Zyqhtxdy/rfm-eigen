"""Where the repository keeps its inputs and its outputs.

``code/data`` holds the recorded output that the builders read; ``figures``
beside ``code`` holds what they write, which is where the manuscript includes
them from.
"""

from __future__ import annotations

import os
from pathlib import Path

#: The ``code`` directory of this repository.
ROOT = Path(__file__).resolve().parents[1]


def _output_root() -> Path | None:
    value = os.environ.get("RFM_REPRO_OUTPUT_ROOT", "").strip()
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


OUTPUT_ROOT = _output_root()
DATA = (OUTPUT_ROOT / "data") if OUTPUT_ROOT else (ROOT / "data")
FIGURES = (OUTPUT_ROOT / "figures") if OUTPUT_ROOT else (ROOT.parent / "figures")
ARTIFACTS = (OUTPUT_ROOT / "artifacts") if OUTPUT_ROOT else (ROOT / "artifacts")
CONFIGS = ROOT / "configs"
EXPERIMENT4_RESULTS = ARTIFACTS / "experiment4"
EXPERIMENT4_CONFIGS = CONFIGS / "experiment4"
REPO_DATA = ROOT / "data"
REPO_ARTIFACTS = ROOT / "artifacts"
REPO_EXPERIMENT4_RESULTS = REPO_ARTIFACTS / "experiment4"
