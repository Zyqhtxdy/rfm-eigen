r"""Assembly of the discrete pencil and the truncated Rayleigh--Ritz solve.

Assembly is the same in every experiment: the features and their gradients are
evaluated at the quadrature points, and the bilinear forms

.. math::

    a(u,v)=\int_\Omega A\nabla u\cdot\nabla v + c\,uv,\qquad
    b(u,v)=\int_\Omega uv

are contracted against the quadrature weights.  Because the features lie in
:math:`H_0^1(\Omega)` by construction, the resulting matrices are those of the
continuous forms restricted to :math:`V_N`, and the min--max characterization
applies pathwise.

The solve needs one further step.  A random basis is badly conditioned -- two
features drawn close together are nearly parallel -- so the pencil is first
rescaled to a unit diagonal and then whitened, and directions below a stated
threshold are discarded before the reduced problem is formed.  The threshold is
fixed in advance and never chosen against a reference value.  Two whitenings
appear in Section 4 and are kept apart here, because they retain different
directions and therefore give different numbers:

``solve_energy_whitened``
    whitens by the energy form and reads the eigenvalues off the reduced mass
    matrix as :math:`\lambda=1/\mu`.  Example 1 uses this route, which is the
    natural one when several eigenvalues at the bottom are wanted at once.

``solve_mass_whitened``
    whitens by the mass form and reads the eigenvalues off the reduced energy
    matrix directly.  Experiments 2 and 3 use this route.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import eigh
from scipy.linalg.blas import dsyrk

#: Threshold floor: directions supported only at roundoff are never retained.
#: Dimension times machine epsilon is the standard scale, with a floor of 128
#: dimensions so that a small test does not keep noise.
MINIMUM_TRUNCATION_DIMENSION = 128


def truncation_tolerance(size: int, dtype: np.dtype | type = np.float64) -> float:
    """The relative threshold below which a whitened direction is discarded.

    Reference-independent by construction: it depends on the matrix size and the
    working precision only, so it is fixed before any reference eigenvalue is
    read.
    """
    return float(
        max(MINIMUM_TRUNCATION_DIMENSION, int(size)) * np.finfo(np.dtype(dtype)).eps
    )


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def assemble_pencil(
    values: np.ndarray,
    gradients: np.ndarray,
    weights: np.ndarray,
    *,
    diffusion: np.ndarray | float = 1.0,
    reaction: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """The energy and mass matrices of the features at the quadrature points.

    ``values`` has shape ``(points, features)`` and ``gradients`` shape
    ``(points, features, dimension)``.  ``diffusion`` and ``reaction`` are the
    coefficients sampled at the same points, or constants.  Both matrices are
    symmetrized before being returned, so that the rounding of the contraction
    cannot break the symmetry the solver assumes.
    """
    mass = values.T @ (weights[:, None] * values)

    if np.isscalar(diffusion) and float(diffusion) == 1.0:
        # A unit coefficient leaves one product per direction, and accumulating
        # them as separate matrix products is both the fastest route and the one
        # with the smallest working set.
        energy = np.zeros_like(mass)
        for axis in range(gradients.shape[2]):
            slice_ = gradients[:, :, axis]
            energy += slice_.T @ (weights[:, None] * slice_)
    else:
        # With a variable coefficient the weight and the coefficient are folded
        # into one contraction over the point and direction axes together.
        energy = np.einsum(
            "q,q,qid,qjd->ij", weights, diffusion, gradients, gradients, optimize=True
        )
    if not (np.isscalar(reaction) and float(reaction) == 0.0):
        energy += values.T @ ((weights * reaction)[:, None] * values)

    return 0.5 * (energy + energy.T), 0.5 * (mass + mass.T)


def _weighted_gram(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    r""":math:`X^\top\mathrm{diag}(w)X` for nonnegative ``w``, as one rank-k update."""
    scaled = matrix * np.sqrt(weights)[:, None]
    upper = dsyrk(1.0, scaled, trans=1, lower=0)
    return upper + np.triu(upper, 1).T


def assemble_cosine_pencil(
    points: np.ndarray,
    weights: np.ndarray,
    omega: np.ndarray,
    phase: np.ndarray,
    factor,
    *,
    diffusion: np.ndarray | float = 1.0,
    reaction: np.ndarray | float = 0.0,
    block: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    r"""The same matrices as :func:`assemble_pencil`, without the gradient array.

    Written out, the gradient of a cosine feature is

    .. math::

        \partial_d\varphi_i(x) = \partial_d\eta(x)\cos_i(x)
                                 - \eta(x)\sin_i(x)\,\omega_{i,d},

    and the energy is the sum over :math:`d` of one Gram matrix in that
    quantity.  Each of those is accumulated with a single symmetric rank-k
    update over blocks of quadrature points, so the largest array in flight is
    one block instead of the whole ``(points, features, dimension)`` gradient --
    which at the sizes of Example 2 is several gigabytes.  It also halves the
    arithmetic, because a symmetric update does half the products of a general
    one.

    The subtraction above is kept inside the block, exactly where the direct
    assembly performs it.  Multiplying it out into four separate Gram matrices
    would be cheaper still, but those terms differ in size by
    :math:`|\omega|^2`, and summing them loses four to five digits to
    cancellation -- enough to move the reported eigenvalue in its sixth digit.
    Keeping the subtraction inside reproduces the direct assembly to round-off.
    """
    dimension = points.shape[1]
    count = omega.shape[0]
    eta, grad_eta = factor(points)
    diffusion = np.broadcast_to(np.asarray(diffusion, dtype=float), weights.shape)
    reaction = np.broadcast_to(np.asarray(reaction, dtype=float), weights.shape)

    eta_squared = eta * eta
    root_weight = np.sqrt(weights * diffusion)
    mass = np.zeros((count, count))
    reaction_matrix = np.zeros_like(mass)
    energy = np.zeros_like(mass)

    for start in range(0, points.shape[0], block):
        stop = min(start + block, points.shape[0])
        argument = points[start:stop] @ omega.T
        argument += phase[None, :]
        cosine = np.cos(argument)
        sine = np.sin(argument)

        mass += _weighted_gram(cosine, weights[start:stop] * eta_squared[start:stop])
        reaction_matrix += _weighted_gram(
            cosine,
            weights[start:stop] * reaction[start:stop] * eta_squared[start:stop],
        )

        scaled_sine = sine * (root_weight[start:stop] * eta[start:stop])[:, None]
        scaled_cosine = cosine * root_weight[start:stop][:, None]
        for axis in range(dimension):
            column = scaled_cosine * grad_eta[start:stop, axis][:, None]
            column -= scaled_sine * omega[None, :, axis]
            upper = dsyrk(1.0, column, trans=1, lower=0)
            energy += upper + np.triu(upper, 1).T

    energy += reaction_matrix
    return energy, mass


def assemble_cosine_pencil_expanded(
    points: np.ndarray,
    weights: np.ndarray,
    omega: np.ndarray,
    phase: np.ndarray,
    factor,
    *,
    reaction: np.ndarray | float = 0.0,
    block: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    r"""The same matrices again, as four Gram matrices instead of one per direction.

    Squaring the gradient of :math:`\eta\cos_i` and integrating gives

    .. math::

        \int |\nabla\eta|^2\cos_i\cos_j
        - \int \eta\,\nabla\eta\cdot(\omega_i\sin_i\cos_j+\omega_j\cos_i\sin_j)
        + (\omega_i\cdot\omega_j)\int \eta^2\sin_i\sin_j,

    which needs only Gram matrices in :math:`\cos` and :math:`\sin` and one
    entrywise product with :math:`\omega_i\cdot\omega_j`.  The cost no longer
    grows with the dimension, which is what makes ten dimensions affordable:
    :func:`assemble_cosine_pencil` would do ten rank-k updates per block where
    this does three.

    The price is cancellation.  The last term is larger than the first by
    :math:`|\omega|^2`, so the difference loses about :math:`\log_{10}|\omega|^2`
    digits.  With the frequency scale of Example 3 that is under two digits
    and harmless; with the frequencies of Example 2 it would be four or five,
    which is why that experiment uses the other form.  Do not use this one
    without checking the size of the frequencies against the accuracy wanted.
    """
    count = omega.shape[0]
    reaction = np.broadcast_to(np.asarray(reaction, dtype=float), weights.shape)
    frequency_gram = omega @ omega.T

    mass = np.zeros((count, count))
    energy = np.zeros_like(mass)
    for start in range(0, points.shape[0], block):
        stop = min(start + block, points.shape[0])
        block_points = points[start:stop]
        block_weights = weights[start:stop]
        eta, grad_eta = factor(block_points)

        argument = block_points @ omega.T + phase
        cosine = np.cos(argument)
        sine = np.sin(argument)
        eta_squared = eta * eta
        grad_squared = np.sum(grad_eta * grad_eta, axis=1)

        mass += cosine.T @ ((block_weights * eta_squared)[:, None] * cosine)
        energy += cosine.T @ (
            (block_weights * (grad_squared + reaction[start:stop] * eta_squared))[
                :, None
            ]
            * cosine
        )
        cross = cosine.T @ (
            ((block_weights * eta)[:, None] * sine) * (grad_eta @ omega.T)
        )
        energy -= cross + cross.T
        energy += (
            sine.T @ ((block_weights * eta_squared)[:, None] * sine)
        ) * frequency_gram

    return 0.5 * (energy + energy.T), 0.5 * (mass + mass.T)


# --------------------------------------------------------------------------
# the truncated solve
# --------------------------------------------------------------------------
@dataclass
class RitzSolution:
    """The eigenvalues, the coefficients and the truncation diagnostics."""

    eigenvalues: np.ndarray
    coefficients: np.ndarray
    retained: int
    tolerance: float
    spectrum: np.ndarray = field(repr=False)
    #: The retained whitened basis, in the coordinates of the original features.
    #: It spans the part of the trial space the solve actually used, which is
    #: what a best-approximation error has to be measured against.
    basis: np.ndarray | None = field(default=None, repr=False)
    #: The eigenvalues of the reduced problem, before they are inverted.  A
    #: criticality problem reports these directly -- they are the multiplication
    #: factor -- and inverting twice would round twice.
    reduced_eigenvalues: np.ndarray | None = field(default=None, repr=False)
    diagnostics: dict[str, Any] = field(default_factory=dict, repr=False)


def _truncation_diagnostics(
    spectrum: np.ndarray, keep: np.ndarray, tolerance: float
) -> dict[str, float]:
    """How safely the threshold separated the retained directions from the rest.

    A small margin means the number reported would move if the threshold moved,
    and is the quantity to watch; the full normalized spectrum is kept alongside
    it so that the decision can be audited without rerunning the solve.
    """
    retained = spectrum[keep]
    removed = spectrum[~keep]
    return {
        "retained": int(np.sum(keep)),
        "tolerance": float(tolerance),
        "margin_relative": float(np.min(np.abs(spectrum - tolerance)) / tolerance),
        "smallest_retained": float(np.min(retained)),
        "largest_removed": float(np.max(removed)) if removed.size else 0.0,
        "retained_condition_number": float(np.max(retained) / np.min(retained)),
    }


def solve_energy_whitened(
    energy: np.ndarray,
    mass: np.ndarray,
    count: int,
    *,
    tolerance: float | None = None,
) -> RitzSolution:
    r"""Whiten by the energy form; used by Example 1.

    The columns are first rescaled so that the energy has a unit diagonal, which
    removes the spread in feature amplitude.  The whitened basis then makes the
    energy the identity, and the ``count`` smallest eigenvalues are the
    reciprocals of the ``count`` largest eigenvalues of the reduced mass matrix.
    Reading them from the largest end is what keeps the small eigenvalues
    accurate, since it is the large mass eigenvalues that are well separated.

    The returned coefficients are normalized in the mass form, so that the Ritz
    functions have unit :math:`L^2` norm.
    """
    if tolerance is None:
        tolerance = truncation_tolerance(energy.shape[0], energy.dtype)

    diagonal = np.diag(energy).copy()
    positive = diagonal > 0.0
    if int(np.sum(positive)) < count:
        raise RuntimeError("too few features carry positive energy")

    scale = 1.0 / np.sqrt(diagonal[positive])
    scaled = (scale[:, None] * energy[np.ix_(positive, positive)]) * scale[None, :]
    spectrum, directions = eigh(0.5 * (scaled + scaled.T))
    spectrum = np.maximum(spectrum, 0.0)

    normalized = spectrum / spectrum[-1]
    keep = spectrum > tolerance * spectrum[-1]
    if int(np.sum(keep)) < count:
        raise RuntimeError("the truncation threshold retained too few directions")

    whitening = directions[:, keep] / np.sqrt(spectrum[keep])[None, :]
    basis = np.zeros((energy.shape[0], int(np.sum(keep))))
    basis[positive, :] = scale[:, None] * whitening

    reduced = basis.T @ mass @ basis
    reduced = 0.5 * (reduced + reduced.T)
    retained = basis.shape[1]
    mass_values, mass_vectors = eigh(
        reduced, subset_by_index=[retained - count, retained - 1]
    )
    order = np.argsort(mass_values)[::-1]
    mass_values = mass_values[order]
    mass_vectors = mass_vectors[:, order]
    if np.any(mass_values <= 0.0):
        raise RuntimeError("nonpositive reduced mass eigenvalue")

    coefficients = basis @ (mass_vectors / np.sqrt(mass_values)[None, :])
    return RitzSolution(
        eigenvalues=1.0 / mass_values,
        coefficients=coefficients,
        retained=retained,
        tolerance=float(tolerance),
        spectrum=normalized,
        basis=basis,
        reduced_eigenvalues=mass_values,
        diagnostics=_truncation_diagnostics(normalized, keep, tolerance),
    )


def solve_mass_whitened(
    energy: np.ndarray,
    mass: np.ndarray,
    count: int,
    *,
    tolerance: float | None = None,
) -> RitzSolution:
    """Whiten by the mass form; used by Experiments 2 and 3.

    The mirror image of the previous routine: the columns are rescaled to a unit
    mass diagonal, the whitened basis makes the mass the identity, and the
    ``count`` smallest eigenvalues are read directly off the reduced energy
    matrix.  The relative residual of the original pencil is reported, so that a
    number can be rejected without reference to an exact value.
    """
    if tolerance is None:
        tolerance = truncation_tolerance(mass.shape[0], mass.dtype)
    if tolerance <= 0.0:
        raise ValueError("the truncation tolerance must be positive")

    diagonal = np.diag(mass).copy()
    positive = diagonal > np.finfo(mass.dtype).tiny
    if not np.any(positive):
        raise RuntimeError("the mass matrix has no positive diagonal entry")

    scale = 1.0 / np.sqrt(diagonal[positive])
    mass_scaled = (scale[:, None] * mass[np.ix_(positive, positive)]) * scale[None, :]
    energy_scaled = (scale[:, None] * energy[np.ix_(positive, positive)]) * scale[
        None, :
    ]
    mass_scaled = 0.5 * (mass_scaled + mass_scaled.T)
    energy_scaled = 0.5 * (energy_scaled + energy_scaled.T)

    spectrum, directions = eigh(mass_scaled, driver="evd", check_finite=False)
    normalized = spectrum / spectrum[-1]
    keep = spectrum > tolerance * spectrum[-1]
    if int(np.sum(keep)) < count:
        raise RuntimeError("the truncation threshold retained too few directions")

    whitening = directions[:, keep] / np.sqrt(spectrum[keep])[None, :]
    reduced = whitening.T @ energy_scaled @ whitening
    values, vectors = eigh(
        0.5 * (reduced + reduced.T),
        subset_by_index=[0, count - 1],
        check_finite=False,
    )

    coefficients = np.zeros((mass.shape[0], count))
    coefficients[positive, :] = scale[:, None] * (whitening @ vectors)

    diagnostics = _truncation_diagnostics(normalized, keep, tolerance)
    diagnostics["generalized_residual"] = _relative_residual(
        energy, mass, values[0], coefficients[:, 0]
    )
    return RitzSolution(
        eigenvalues=values,
        coefficients=coefficients,
        retained=int(np.sum(keep)),
        tolerance=float(tolerance),
        spectrum=normalized,
        diagnostics=diagnostics,
    )


def _relative_residual(
    energy: np.ndarray, mass: np.ndarray, value: float, vector: np.ndarray
) -> float:
    r""":math:`\|Kc-\lambda Mc\|` against the size of its two terms.

    Normalizing by the two terms rather than by either one keeps the quantity
    meaningful when the eigenvalue is small, which is the regime the experiments
    work in.
    """
    stiff = energy @ vector
    weighted = mass @ vector
    denominator = np.linalg.norm(stiff) + abs(float(value)) * np.linalg.norm(weighted)
    residual = stiff - float(value) * weighted
    return float(np.linalg.norm(residual) / max(denominator, np.finfo(float).tiny))


def pencil_residuals(
    energy: np.ndarray,
    mass: np.ndarray,
    eigenvalues: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """The relative residual of every computed pair, in the original basis.

    Computed after the whitening has been undone, so it certifies the numbers
    that are reported rather than the numbers the reduced problem produced.
    """
    return np.asarray(
        [
            _relative_residual(energy, mass, float(value), coefficient)
            for value, coefficient in zip(eigenvalues, coefficients.T, strict=True)
        ]
    )


def truncation_sensitivity(
    energy: np.ndarray,
    mass: np.ndarray,
    count: int,
    *,
    multipliers: tuple[float, ...] = (0.25, 1.0, 4.0),
) -> list[dict[str, float]]:
    """Repeat the mass-whitened solve at multiples of the stated threshold.

    Registered in advance and never given a reference value: the point is to
    show that the reported eigenvalue does not depend on where the threshold was
    put, not to select a threshold.
    """
    base = truncation_tolerance(mass.shape[0], mass.dtype)
    rows: list[dict[str, float]] = []
    for multiplier in multipliers:
        solution = solve_mass_whitened(
            energy, mass, count, tolerance=base * float(multiplier)
        )
        rows.append(
            {
                "multiplier": float(multiplier),
                "tolerance": base * float(multiplier),
                "retained": float(solution.retained),
                "eigenvalue": float(solution.eigenvalues[0]),
                "generalized_residual": float(
                    solution.diagnostics["generalized_residual"]
                ),
            }
        )
    return rows


def dominant_generalized_eigenpair(
    fission: np.ndarray, whitening: np.ndarray, *, tolerance: float = 0.0
) -> tuple[float, np.ndarray]:
    r"""The largest eigenpair of :math:`W^\top F W`, without forming that matrix.

    A criticality problem wants one number -- the multiplication factor is the
    dominant eigenvalue -- and forming the reduced matrix to get it costs two
    products of the full whitening and a complete tridiagonalization.  The
    operator :math:`x\mapsto W^\top(F(Wx))` needs three matrix--vector products
    per step instead, and Lanczos on it converges quickly because the
    fundamental mode of a criticality problem is well separated.

    The reduced route is still the default in Example 4, because it is what
    the recorded values were produced with; this is the faster route and the two
    agree to the working precision, which the tests assert.  The device
    counterpart in :mod:`rfmeig.gpu` uses power iteration for the same reason and
    carries a warning about its stopping test that applies here in weaker form.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    size = whitening.shape[1]

    def apply(vector: np.ndarray) -> np.ndarray:
        return whitening.T @ (fission @ (whitening @ vector))

    operator = LinearOperator((size, size), matvec=apply, dtype=float)
    values, vectors = eigsh(operator, k=1, which="LA", tol=tolerance, maxiter=5000)
    return float(values[0]), vectors[:, 0]
