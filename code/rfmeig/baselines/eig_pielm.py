"""Eig-PIELM, the fixed-feature collocation baseline of Example 2.

Reproduced as published in Mishra et al., CMAME 451 (2026) 118674, rather than
paraphrased:

* the basis is a tensor product of **Bernstein polynomials** on the bounding
  box, ``n+1`` per axis for degree ``n``, chosen there for their conditioning
  and partition-of-unity properties -- not a random activation;
* the boundary condition is imposed **exactly at the boundary collocation set**
  by reparameterizing ``beta = T_b y`` with ``T_b`` a basis of the null space of
  the boundary block, which is what the paper calls the boundary-admissible
  subspace;
* only the interior residuals enter the loss, giving ``A``, ``S`` and ``G`` as
  in its equation (17), and in the admissible coordinates the skew part of ``S``
  vanishes, so its equation (19) is the **symmetric** generalized problem
  ``S_red y = lambda G_red y`` with ``G_red`` positive definite.

Two consequences matter for the comparison the manuscript draws.  The basis is
deterministic, so unlike the sampled space this baseline has no draw-to-draw
variation and nothing to average over; the table therefore reports a single
value per setting.  And the admissible subspace has dimension
``N_phi - rank(Phi_boundary)``, so its effective number of unknowns is strictly
smaller than its number of basis functions -- which is the dimension column of
Table 2.

The boundary condition here holds **at the collocation points**, algebraically.
Section 2.3's construction instead holds in the continuous trace, for every
realization, which is what lets min-max and conforming spectral approximation
apply pathwise.  The manuscript keeps those two senses apart deliberately.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.linalg
from scipy.special import comb


@dataclass
class Result:
    values: np.ndarray
    dimension: int
    admissible: int
    seconds: float
    assemble_seconds: float
    solve_seconds: float


def _bernstein(t: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Values and the first two derivatives of the Bernstein basis on ``[0,1]``.

    The derivative identities of the basis are used rather than a numerical
    difference, so the second derivative is exact for every degree.
    """
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)

    def raw(order: int) -> np.ndarray:
        if order < 0:
            return np.zeros((t.size, 0))
        indices = np.arange(order + 1)
        coefficients = comb(order, indices)
        return (
            coefficients
            * t[:, None] ** indices
            * (1.0 - t)[:, None] ** (order - indices)
        )

    value = raw(degree)
    lower = raw(degree - 1)
    lower2 = raw(degree - 2)

    first = np.zeros_like(value)
    if degree >= 1:
        padded = np.zeros((t.size, degree + 2))
        padded[:, 1 : degree + 1] = lower
        first = degree * (padded[:, :-1] - padded[:, 1:])[:, : degree + 1]

    second = np.zeros_like(value)
    if degree >= 2:
        padded = np.zeros((t.size, degree + 3))
        padded[:, 2 : degree + 1] = lower2
        second = (
            degree
            * (degree - 1)
            * (padded[:, :-2] - 2.0 * padded[:, 1:-1] + padded[:, 2:])[:, : degree + 1]
        )
    return value, first, second


def _tensor_basis(points: np.ndarray, degree: int, lower, upper):
    """Tensor-product Bernstein values and negative Laplacian on a box."""
    dimension = points.shape[1]
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    span = upper - lower
    scaled = (points - lower) / span

    per_axis = [_bernstein(scaled[:, axis], degree) for axis in range(dimension)]

    values = per_axis[0][0]
    for axis in range(1, dimension):
        values = (values[:, :, None] * per_axis[axis][0][:, None, :]).reshape(
            points.shape[0], -1
        )

    laplacian = np.zeros_like(values)
    for target in range(dimension):
        block = None
        for axis in range(dimension):
            factor = (
                per_axis[axis][2] / span[axis] ** 2
                if axis == target
                else per_axis[axis][0]
            )
            block = (
                factor
                if block is None
                else (block[:, :, None] * factor[:, None, :]).reshape(
                    points.shape[0], -1
                )
            )
        laplacian += block

    gradient = []
    for target in range(dimension):
        block = None
        for axis in range(dimension):
            factor = (
                per_axis[axis][1] / span[axis]
                if axis == target
                else per_axis[axis][0]
            )
            block = (
                factor
                if block is None
                else (block[:, :, None] * factor[:, None, :]).reshape(
                    points.shape[0], -1
                )
            )
        gradient.append(block)
    return values, -laplacian, np.stack(gradient, axis=0)


def solve(
    domain,
    degree: int,
    seed: int = 0,
    grid_per_axis: int | None = None,
    boundary_points: int | None = None,
    count: int = 6,
    boundary_tolerance: float = 1.0e-10,
    mass_tolerance: float = 1.0e-11,
) -> Result:
    """One Eig-PIELM solve at the given polynomial degree."""
    rng = np.random.default_rng(seed)
    functions = (degree + 1) ** domain.dimension
    # the admissible subspace needs room: fewer boundary conditions than unknowns
    if boundary_points is None:
        # The paper prescribes the minimal sampling that determines the trace,
        # and warns against oversampling -- but that rule is stated for
        # rectangles, where the trace on an edge is a one-dimensional polynomial
        # of known degree.  On a curved boundary the trace of a tensor polynomial
        # is not a polynomial in any boundary parameter, so the minimal count
        # does not determine it, and sampling only that many leaves the
        # admissible space under-constrained: measured on the ball, it costs the
        # baseline an order of magnitude.  Sampling until the rank saturates is
        # the faithful generalization, and it is what the baseline is given here.
        boundary_points = 12 * (degree + 1) ** (domain.dimension - 1)
    if grid_per_axis is None:
        grid_per_axis = 10 * (degree + 1) if domain.dimension == 2 else 3 * (degree + 1)

    started = time.perf_counter()
    interior = domain.structured_interior(grid_per_axis)
    boundary = domain.sample_boundary(boundary_points, rng)

    lower = interior.min(axis=0)
    upper = interior.max(axis=0)
    for block in (boundary,):
        lower = np.minimum(lower, block.min(axis=0))
        upper = np.maximum(upper, block.max(axis=0))
    pad = 1.0e-12 * np.maximum(np.abs(upper - lower), 1.0)
    lower, upper = lower - pad, upper + pad

    values, negative_laplacian, gradient = _tensor_basis(
        interior, degree, lower, upper
    )
    boundary_values, _, _ = _tensor_basis(boundary, degree, lower, upper)

    # In divergence form the residual carries a first-order term as well:
    # -div(a grad u) = -a Lap u - grad a . grad u.  With a constant the second
    # term vanishes and this is the plain Laplacian the baseline started from.
    conductivity = domain.coefficient(interior)
    negative_laplacian = conductivity[:, None] * negative_laplacian
    conductivity_gradient = domain.coefficient_gradient(interior)
    for axis in range(interior.shape[1]):
        negative_laplacian -= conductivity_gradient[:, axis][:, None] * gradient[axis]

    # the boundary-admissible subspace: exactly the null space of the boundary rows
    # the boundary block is tall; only its right factor is needed, and asking
    # for the full left factor would build an enormous matrix for nothing
    _, singular, right = np.linalg.svd(boundary_values, full_matrices=False)
    rank = int((singular > boundary_tolerance * max(singular[0], 1.0e-300)).sum())
    admissible_map = right[rank:].T
    if admissible_map.shape[1] < count + 1:
        raise RuntimeError(
            f"admissible subspace of dimension {admissible_map.shape[1]} "
            f"cannot carry {count} modes"
        )

    # In admissible coordinates the two design matrices are Gram matrices, so
    # forming them squares the conditioning of a high-degree Bernstein basis and
    # the mass matrix stops being positive definite well before the polynomial
    # space stops being useful.  Whitening on the singular values of the design
    # matrix itself, and dropping the directions the data cannot support, is the
    # standard remedy; it changes the arithmetic, not the method.
    interior_block = values @ admissible_map
    operator_block = negative_laplacian @ admissible_map
    left_vectors, singular, right_vectors = np.linalg.svd(
        interior_block, full_matrices=False
    )
    keep = singular > mass_tolerance * max(singular[0], 1.0e-300)
    whitening = right_vectors[keep].T / singular[keep]
    reduced_stiffness = (operator_block @ whitening).T @ (interior_block @ whitening)
    reduced_stiffness = 0.5 * (reduced_stiffness + reduced_stiffness.T)
    assembled = time.perf_counter()

    eigenvalues = scipy.linalg.eigvalsh(reduced_stiffness)
    positive = eigenvalues[eigenvalues > 1.0e-8]
    chosen = np.sort(positive)[:count]
    finished = time.perf_counter()
    return Result(
        values=np.asarray(chosen),
        dimension=functions,
        admissible=int(keep.sum()),
        seconds=finished - started,
        assemble_seconds=assembled - started,
        solve_seconds=finished - assembled,
    )
