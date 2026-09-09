r"""The quantities Section 4 reports, and how each one is computed.

Three of them need care.

*The eigenspace gap.*  The target eigenspace and its Ritz approximation have the
same dimension, so the gap between them is the sine of the largest principal
angle.  It is obtained from the singular values of the cross Gram matrix taken
between the two bases after each has been made orthonormal in
:math:`H^1(\Omega)`; the smallest singular value is the cosine of the largest
angle.  Because the quantity is a sine of an angle between subspaces it is
insensitive to the basis chosen inside either one, which is exactly what an
eigenvalue of multiplicity greater than one requires.

*The best-approximation error.*  The same construction with the projection onto
the whole trial space in place of the Ritz subspace: it measures how well the
random features could approximate the target, independently of how well the
Rayleigh--Ritz method actually does.

*The rate.* Fitted by least squares in log--log coordinates to the chosen
statistic across draws, with a stratified bootstrap interval. Example 1 uses
the arithmetic mean over all feature counts; quantile fits remain available
for comparison with the archived analysis.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


# --------------------------------------------------------------------------
# Gram matrices in the energy and the graph norm
# --------------------------------------------------------------------------
def h1_gram(
    weights: np.ndarray,
    left_values: np.ndarray,
    left_gradients: np.ndarray,
    right_values: np.ndarray,
    right_gradients: np.ndarray,
) -> np.ndarray:
    r"""The matrix of :math:`(u,v)_{H^1}=\int uv+\nabla u\cdot\nabla v`."""
    gram = left_values.T @ (weights[:, None] * right_values)
    gram += np.einsum(
        "q,qid,qjd->ij", weights, left_gradients, right_gradients, optimize=True
    )
    return gram


def _inverse_square_root(gram: np.ndarray) -> np.ndarray:
    """:math:`G^{-1/2}` of a symmetric positive definite Gram matrix."""
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
    return vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T


def eigenspace_gap(
    weights: np.ndarray,
    exact_values: np.ndarray,
    exact_gradients: np.ndarray,
    ritz_values: np.ndarray,
    ritz_gradients: np.ndarray,
) -> float:
    r"""The :math:`H^1` gap between two subspaces of equal dimension.

    Both bases are made orthonormal by their own inverse square root, and the
    singular values of the resulting cross Gram matrix are the cosines of the
    principal angles.  The gap is the sine of the largest of them, so it is one
    when the two subspaces are orthogonal and zero when they coincide.
    """
    exact_gram = h1_gram(
        weights, exact_values, exact_gradients, exact_values, exact_gradients
    )
    ritz_gram = h1_gram(
        weights, ritz_values, ritz_gradients, ritz_values, ritz_gradients
    )
    cross = h1_gram(weights, exact_values, exact_gradients, ritz_values, ritz_gradients)
    return eigenspace_gap_from_grams(exact_gram, ritz_gram, cross)


def eigenspace_gap_from_grams(
    exact_gram: np.ndarray, ritz_gram: np.ndarray, cross: np.ndarray
) -> float:
    """The same principal-angle distance from already assembled H1 Gram matrices."""
    aligned = _inverse_square_root(exact_gram) @ cross @ _inverse_square_root(ritz_gram)
    cosines = np.linalg.svd(aligned, compute_uv=False)
    smallest = float(np.clip(np.min(cosines), 0.0, 1.0))
    return math.sqrt(max(0.0, 1.0 - smallest**2))


def best_approximation_error(
    weights: np.ndarray,
    exact_values: np.ndarray,
    exact_gradients: np.ndarray,
    trial_values: np.ndarray,
    trial_gradients: np.ndarray,
    *,
    relative_tolerance: float = 1.0e-12,
) -> float:
    r"""How well the trial space approximates the target subspace in :math:`H^1`.

    The projection onto the trial space is formed in the :math:`H^1` inner
    product, discarding trial directions the Gram matrix cannot resolve, and the
    residual is measured in the norm of the target.  The result is the sine of
    the largest angle between the target and its projection, so it lies in
    :math:`[0,1]` and is directly comparable with :func:`eigenspace_gap`.
    """
    exact_gram = h1_gram(
        weights, exact_values, exact_gradients, exact_values, exact_gradients
    )
    trial_gram = h1_gram(
        weights, trial_values, trial_gradients, trial_values, trial_gradients
    )
    cross = h1_gram(
        weights, exact_values, exact_gradients, trial_values, trial_gradients
    )
    return best_approximation_from_grams(
        exact_gram, trial_gram, cross, relative_tolerance=relative_tolerance
    )


def best_approximation_from_grams(
    exact_gram: np.ndarray,
    trial_gram: np.ndarray,
    cross: np.ndarray,
    *,
    relative_tolerance: float = 1.0e-12,
) -> float:
    """The H1 projection error, using the same normalization as field evaluation."""
    exact_gram = 0.5 * (exact_gram + exact_gram.T)
    trial_gram = 0.5 * (trial_gram + trial_gram.T)

    trial_values_, trial_vectors = np.linalg.eigh(trial_gram)
    keep = trial_values_ > relative_tolerance * trial_values_[-1]
    if trial_values_[-1] <= 0.0 or not np.any(keep):
        projection = np.zeros_like(exact_gram)
    else:
        reduced_cross = cross @ trial_vectors[:, keep]
        projection = (reduced_cross / trial_values_[keep]) @ reduced_cross.T

    residual = exact_gram - projection
    residual = 0.5 * (residual + residual.T)

    exact_spectrum, exact_vectors = np.linalg.eigh(exact_gram)
    resolved = exact_spectrum > relative_tolerance * exact_spectrum[-1]
    normalizer = exact_vectors[:, resolved] @ np.diag(
        1.0 / np.sqrt(exact_spectrum[resolved])
    )
    reduced = normalizer.T @ residual @ normalizer
    largest = float(np.max(np.linalg.eigvalsh(0.5 * (reduced + reduced.T))))
    return math.sqrt(float(np.clip(largest, 0.0, 1.0)))


# --------------------------------------------------------------------------
# eigenvalue errors within and around a cluster
# --------------------------------------------------------------------------
def cluster_errors(
    eigenvalues: np.ndarray, first: int, last: int, exact: float
) -> dict[str, float]:
    """Relative errors of a cluster occupying positions ``first`` to ``last``.

    Positions are one-based, as they are in the manuscript.  Four numbers are
    reported: how far the cluster sits from the exact level, how far its members
    have split from each other, and the two gaps that separate it from the level
    below and the level above.  The last two are what decide whether the cluster
    has been identified at all, and they are recorded as such rather than being
    checked silently.
    """
    inside = eigenvalues[first - 1 : last]
    return {
        "cluster_relative_error": float(np.max(np.abs(inside - exact)) / exact),
        "splitting_relative_error": float(
            (eigenvalues[last - 1] - eigenvalues[first - 1]) / exact
        ),
        "lower_external_gap_relative": float(
            (eigenvalues[first - 1] - eigenvalues[first - 2]) / exact
        ),
        "upper_external_gap_relative": float(
            (eigenvalues[last] - eigenvalues[last - 1]) / exact
        ),
    }


def relative_error(value: float, reference: float) -> float:
    """The relative error against a reference value, which must not vanish."""
    return float(abs(float(value) - float(reference)) / abs(float(reference)))


# --------------------------------------------------------------------------
# rates
# --------------------------------------------------------------------------
def regression_slope(sizes: Sequence[float], values: Sequence[float]) -> float:
    """Least squares slope of ``log(values)`` against ``log(sizes)``."""
    return float(
        np.polyfit(
            np.log(np.asarray(sizes, dtype=float)),
            np.log(np.asarray(values, dtype=float)),
            1,
        )[0]
    )


def bootstrap_slope_interval(
    trials: Sequence[dict[str, Any]],
    sizes: Sequence[int],
    field: str,
    *,
    level: float = 0.9,
    replicates: int = 4000,
    base_seed: int = 2031071401,
    statistic: str = "quantile",
) -> tuple[float, float]:
    """A 95 percent interval for the fitted rate, resampling the draws.

    At each feature count the draws are resampled with replacement, the upper
    quantile is recomputed, and a slope is fitted through the resampled
    quantiles.  The interval is the central 95 percent of those slopes.  The seed
    is derived from the field name, so that two quantities fitted in the same run
    do not share a resampling pattern.
    """
    if statistic not in {"mean", "quantile"}:
        raise ValueError(f"unknown statistic: {statistic}")
    if statistic == "mean":
        # Pair resamples across error fields, as in the refined Example 1 study.
        rng = np.random.default_rng(base_seed)
        x = np.log(np.asarray(sizes, dtype=float))
        weights = (x - x.mean()) / np.sum((x - x.mean()) ** 2)
        means = []
        for size in sizes:
            sample = np.asarray([float(row[field]) for row in trials if int(row["N"]) == size])
            indices = rng.integers(0, len(sample), (replicates, len(sample)))
            means.append(sample[indices].mean(axis=1))
        slopes = np.log(np.column_stack(means)) @ weights
        return tuple(float(v) for v in np.quantile(slopes, [.025, .975]))
    rng = np.random.default_rng(base_seed + sum(ord(char) for char in field))
    grouped = {
        size: np.asarray([float(row[field]) for row in trials if int(row["N"]) == size])
        for size in sizes
    }
    axis = np.asarray(sizes, dtype=float)
    slopes = np.empty(replicates)
    for replicate in range(replicates):
        quantiles = [
            float(
                np.quantile(
                    rng.choice(grouped[size], size=grouped[size].size, replace=True),
                    level,
                )
            )
            for size in sizes
        ]
        slopes[replicate] = regression_slope(axis, quantiles)
    return float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975))
