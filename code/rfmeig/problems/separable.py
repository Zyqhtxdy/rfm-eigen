r"""Separable Schrödinger operators, and the reference values they supply.

Experiments 1 and 3 both take a potential that is a sum of one-dimensional
terms,

.. math::

    -\Delta u + \Big(\sum_{i=1}^d V(x_i)\Big) u = \lambda u
    \quad\text{in }(0,1)^d,\qquad u=0\text{ on the boundary},

so that every eigenvalue is a sum of one-dimensional eigenvalues and every
eigenfunction a product of one-dimensional eigenfunctions.  That is what makes
the reference value exact to the precision of a one-dimensional solve, in a
dimension where no discretization of the full problem could supply one.  It also
produces multiplicity for free: whenever two one-dimensional levels differ, the
two orders of the same pair give the same sum, and the corresponding
eigenvalue of the :math:`d`-dimensional problem has multiplicity at least two.
That eigenvalue is the target of Example 1.

The one-dimensional problem is solved in a sine basis, which is the eigenbasis
of the Laplacian with the same boundary condition, so the kinetic term is
diagonal and only the potential has to be integrated.  The potential of
Example 1 has two kinks; the composite rule of :mod:`rfmeig.quadrature` puts
a cell boundary at each of them, so the integration keeps its full order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from rfmeig.quadrature import composite_gauss_unit


# --------------------------------------------------------------------------
# the sine basis
# --------------------------------------------------------------------------
def sine_basis(nodes: np.ndarray, modes: int) -> np.ndarray:
    r""":math:`\sqrt2\sin(k\pi t)`, :math:`k=1,\dots,m`, unit in :math:`L^2`."""
    orders = np.arange(1, modes + 1)
    return math.sqrt(2.0) * np.sin(math.pi * nodes[:, None] * orders[None, :])


def sine_basis_derivative(nodes: np.ndarray, modes: int) -> np.ndarray:
    """The derivative of the same basis, needed for the gradients of the products."""
    orders = np.arange(1, modes + 1)
    return (
        math.sqrt(2.0)
        * (math.pi * orders[None, :])
        * np.cos(math.pi * nodes[:, None] * orders[None, :])
    )


# --------------------------------------------------------------------------
# the potential of Example 1
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DoubleKinkPotential:
    r""":math:`V(t)=2+a(|t-p|+0.7\,|t-(1-p)|)`, the potential of Example 1.

    Two kinks, placed asymmetrically and weighted unequally, so that the
    potential is Lipschitz but not differentiable and the two kinks do not
    coincide.  The eigenfunctions are therefore only :math:`H^{3}` across each
    kink, which is what makes the experiment a test of the source condition
    rather than of a smooth best case.
    """

    amplitude: float = 20.0
    location: float = 0.37
    offset: float = 2.0
    second_weight: float = 0.7

    def __call__(self, t: np.ndarray) -> np.ndarray:
        shape = np.abs(t - self.location) + self.second_weight * np.abs(
            t - (1.0 - self.location)
        )
        return self.offset + self.amplitude * shape

    @property
    def breakpoints(self) -> list[float]:
        """Where the potential is not differentiable, in increasing order."""
        return sorted(
            value for value in (self.location, 1.0 - self.location) if 0.0 < value < 1.0
        )

    def describe(self) -> str:
        """The formula, as it is quoted in the manuscript."""
        return (
            f"V(t)={self.offset:g}+{self.amplitude:g}"
            f"(|t-{self.location:g}|+{self.second_weight:g}"
            f"|t-{1.0 - self.location:g}|)"
        )


# --------------------------------------------------------------------------
# the one-dimensional reference solve
# --------------------------------------------------------------------------
def solve_one_dimensional(
    potential: DoubleKinkPotential, *, modes: int = 112, order: int = 96
) -> tuple[np.ndarray, np.ndarray]:
    r"""Eigenvalues and sine coefficients of :math:`-u''+Vu=\lambda u`.

    The kinetic term is :math:`\mathrm{diag}((k\pi)^2)` because the basis
    diagonalizes it; only the potential is integrated, on a composite rule split
    at the kinks.  ``modes`` and ``order`` are set well beyond the accuracy the
    experiment needs, so that the reference is not what limits the reported
    errors.
    """
    nodes, weights = composite_gauss_unit(potential.breakpoints, order)
    basis = sine_basis(nodes, modes)
    potential_matrix = basis.T @ ((weights * potential(nodes))[:, None] * basis)
    kinetic = np.diag((math.pi * np.arange(1, modes + 1)) ** 2)
    matrix = kinetic + potential_matrix
    return eigh(0.5 * (matrix + matrix.T), driver="evd", check_finite=False)


# --------------------------------------------------------------------------
# products of one-dimensional levels
# --------------------------------------------------------------------------
def sorted_pair_sums(values: np.ndarray, limit: int) -> list[tuple[float, int, int]]:
    """Every sum of two one-dimensional levels below ``limit``, in increasing order.

    The list is over ordered pairs, so a sum formed by two different levels
    appears twice; that repetition is the multiplicity of the two-dimensional
    eigenvalue and is deliberately not removed.
    """
    sums = [
        (float(values[i] + values[j]), i, j) for i in range(limit) for j in range(limit)
    ]
    sums.sort(key=lambda item: item[0])
    return sums


def locate_pair(
    values: np.ndarray,
    pair: tuple[int, int],
    *,
    limit: int = 18,
    tolerance: float = 1.0e-9,
) -> tuple[int, int, float]:
    """Where a given pair of one-dimensional levels sits in the ordered spectrum.

    Returns the first and last one-based positions the sum occupies, and the sum
    itself.  Two distinct positions mean the eigenvalue is multiple, which is the
    situation the experiment is built around.
    """
    sums = sorted_pair_sums(values, limit)
    target = float(values[pair[0]] + values[pair[1]])
    positions = [
        position
        for position, (value, _, _) in enumerate(sums, start=1)
        if abs(value - target) < tolerance
    ]
    if not positions:
        raise RuntimeError("the requested pair does not appear in the ordered spectrum")
    return min(positions), max(positions), target


def leading_pairs(
    values: np.ndarray, count: int, *, limit: int = 18
) -> list[tuple[int, int]]:
    """The ``count`` lowest ordered pairs, which index the lower spectral subspace.

    Example 1 uses these to measure how well the trial space approximates the
    whole subspace up to and including the target level, not the target
    eigenspace alone.
    """
    return [(i, j) for _, i, j in sorted_pair_sums(values, limit)[:count]]


def product_fields(
    points: np.ndarray,
    vectors: np.ndarray,
    pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Values and gradients of the product eigenfunctions at the given points.

    Each product is :math:`\\psi_i(x)\\psi_j(y)` with the one-dimensional factors
    reconstructed from their sine coefficients, so the fields are exact up to the
    truncation of the reference solve.
    """
    modes = vectors.shape[0]
    basis_x = sine_basis(points[:, 0], modes)
    basis_y = sine_basis(points[:, 1], modes)
    derivative_x = sine_basis_derivative(points[:, 0], modes)
    derivative_y = sine_basis_derivative(points[:, 1], modes)

    values: list[np.ndarray] = []
    gradients: list[np.ndarray] = []
    for first, second in pairs:
        psi_x = basis_x @ vectors[:, first]
        psi_y = basis_y @ vectors[:, second]
        d_psi_x = derivative_x @ vectors[:, first]
        d_psi_y = derivative_y @ vectors[:, second]
        values.append(psi_x * psi_y)
        gradients.append(np.column_stack([d_psi_x * psi_y, psi_x * d_psi_y]))
    return np.column_stack(values), np.stack(gradients, axis=1)


def separable_reaction(
    points: np.ndarray, potential: DoubleKinkPotential
) -> np.ndarray:
    """:math:`c(x)=\\sum_i V(x_i)` sampled at the quadrature points."""
    return sum(potential(points[:, axis]) for axis in range(points.shape[1]))
