from __future__ import annotations

import numpy as np


def relative_keff_error(keff: float, reference: float) -> float:
    return abs(keff - reference) / abs(reference)


def mean_power_normalize(values: np.ndarray) -> np.ndarray:
    """Apply the discrete mean-power normalization of Yang et al., Eq. (42)."""
    normalized = np.asarray(values, dtype=float).copy()
    mean = float(np.mean(normalized))
    if abs(mean) <= 1.0e-30:
        raise ValueError("Cannot normalize a scalar flux with zero discrete mean.")
    return normalized / mean


def relative_flux_max_error(approximation: np.ndarray, reference: np.ndarray) -> float:
    """Evaluate the normalized max error in Yang et al., Eqs. (37) and (42)."""
    approximation = mean_power_normalize(approximation)
    reference = mean_power_normalize(reference)
    denominator = max(float(np.max(np.abs(reference))), 1.0e-30)
    return float(np.max(np.abs(approximation - reference)) / denominator)
