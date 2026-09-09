r"""The radially graded unit ball of Example 2, and its exact spectrum.

The problem is Section 2.1's :eq:`model_eigenproblem` with
:math:`A(x)=a(|x|)I` and :math:`c\equiv0`,

.. math::
    -\nabla\cdot\bigl(a(|x|)\nabla u\bigr)=\lambda u\ \text{in }\Omega,
    \qquad u|_{\partial\Omega}=0,
    \qquad a(r)=1+\gamma r^2 ,

on :math:`\Omega=\{x\in\mathbb R^3:|x|<1\}`.  The parabolic profile is the
standard grading of a radially stratified medium; Assumption 2.1 holds with
:math:`\alpha_A=1` and :math:`\beta_A=1+\gamma`.

The grading is radial on purpose.  A coefficient that broke the rotational
symmetry would split the very levels the example exists to resolve, whereas a
radial one leaves the level of angular order :math:`\ell` with the exact
multiplicity :math:`2\ell+1`.  At :math:`\gamma=3` the first ten ordered levels
group as :math:`1+3+1+5`, so the threefold and fivefold clusters of the
manuscript have to be located across an intervening simple level that sits
within 2.4 percent of the fivefold one.

The exact levels are not closed form here, as they are for the Laplacian, but
neither are they extrapolated.  Separation of variables turns them into
one-dimensional radial eigenvalue problems, which :func:`radial_levels` solves
directly; :mod:`rfmeig.tests` pins that solver against the closed-form
constant-coefficient case.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg


def _radial_basis(nodes: np.ndarray, size: int, vanishes_at_zero: bool):
    r"""Legendre polynomials times the factor carrying the end conditions.

    ``R(1)=0`` always; ``R(0)=0`` as well for every angular order above zero,
    because :math:`R(r)Y_\ell^m` is otherwise not in :math:`H^1(\Omega)`.  The
    weak form below does not see that second condition -- its energy stays
    finite without it -- so it is built into the basis instead.
    """
    shifted = 2.0 * nodes - 1.0
    columns, derivatives = [], []
    if vanishes_at_zero:
        factor = nodes * (1.0 - nodes)
        factor_prime = 1.0 - 2.0 * nodes
    else:
        factor = 1.0 - nodes
        factor_prime = -np.ones_like(nodes)

    previous = np.zeros_like(nodes)
    current = np.ones_like(nodes)
    previous_prime = np.zeros_like(nodes)
    current_prime = np.zeros_like(nodes)
    for degree in range(size):
        columns.append(factor * current)
        derivatives.append(factor_prime * current + factor * current_prime)
        # Legendre recurrence in the shifted variable, derivative alongside
        nxt = ((2 * degree + 1) * shifted * current - degree * previous) / (degree + 1)
        nxt_prime = (
            (2 * degree + 1) * (2.0 * current + shifted * current_prime)
            - degree * previous_prime
        ) / (degree + 1)
        previous, current = current, nxt
        previous_prime, current_prime = current_prime, nxt_prime
    return np.array(columns).T, np.array(derivatives).T


def radial_levels(
    coefficient,
    orders: int = 12,
    per_order: int = 6,
    size: int = 32,
    quadrature: int = 400,
) -> list[tuple[float, int]]:
    r"""Exact levels of the graded ball, by separation of variables.

    With :math:`u=R(r)Y_\ell^m` the radial factor solves

    .. math::
        \int_0^1\bigl(a r^2R'S'+\ell(\ell+1)aRS\bigr)\,dr
        =\lambda\int_0^1 r^2RS\,dr ,

    which is the weak problem printed in Example 2.  The :math:`r^{-2}` of the
    angular term cancels against the volume weight, so nothing is singular at
    the origin.  Each angular order contributes its radial eigenvalues with
    multiplicity :math:`2\ell+1`.

    ``size`` is pinned inside the accuracy plateau.  The Legendre columns lose
    orthogonality as the basis grows, so the levels improve to about 1e-12 near
    thirty and then decay again; sizes between 16 and 40 agree to 1.7e-12.
    """
    gauss, weights = np.polynomial.legendre.leggauss(quadrature)
    nodes = 0.5 * (gauss + 1.0)
    weights = 0.5 * weights
    a = np.asarray(coefficient(nodes), dtype=float)

    found: list[tuple[float, int]] = []
    for order in range(orders):
        columns, derivatives = _radial_basis(nodes, size, vanishes_at_zero=order > 0)
        volume = weights * nodes**2
        stiffness = (derivatives * (volume * a)[:, None]).T @ derivatives
        stiffness += (
            columns * (weights * order * (order + 1) * a)[:, None]
        ).T @ columns
        mass = (columns * volume[:, None]).T @ columns
        values = scipy.linalg.eigvalsh(
            0.5 * (stiffness + stiffness.T),
            0.5 * (mass + mass.T),
            subset_by_index=(0, min(per_order, size) - 1),
        )
        found.extend((float(value), 2 * order + 1) for value in values)
    found.sort(key=lambda item: item[0])
    return found


class GradedBall:
    r"""The unit ball with the diffusion coefficient :math:`a(r)=1+\gamma r^2`.

    Carries the three things a method needs from the geometry: a smooth
    function vanishing exactly on the sphere, so that the sampled features lie
    in :math:`H_0^1(\Omega)` for every realization (Section 2.3); a quadrature
    rule the trial space is integrated on; and the coefficient itself.
    """

    dimension = 3
    name = "graded_ball"

    #: The grading ratio a(R)/a(0).  Four to one is a genuine variable
    #: coefficient and is what Example 2 reports.  The sampled features carry
    #: one frequency scale while the local wavenumber goes as 1/sqrt(a), so a
    #: far stronger grading costs every fixed-scale method accuracy: measured
    #: here, ten to one puts the sampled space and the mesh within a factor of
    #: three of each other, and four to one does not.
    DEFAULT_CONTRAST = 3.0

    def __init__(self, radius: float = 1.0, contrast: float = DEFAULT_CONTRAST):
        self.radius = float(radius)
        self.contrast = float(contrast)

    # -- the operator ------------------------------------------------------

    def profile(self, r):
        """The coefficient as a function of the radius alone."""
        return 1.0 + self.contrast * (r / self.radius) ** 2

    def coefficient(self, points: np.ndarray) -> np.ndarray:
        return self.profile(np.linalg.norm(points, axis=1))

    def coefficient_gradient(self, points: np.ndarray) -> np.ndarray:
        """:math:`\\nabla a`, which only the collocation baseline needs.

        In divergence form the strong residual carries a first-order term,
        :math:`-\\nabla\\cdot(a\\nabla u)=-a\\Delta u-\\nabla a\\cdot\\nabla u`,
        and Eig-PIELM has to drive both to zero at its collocation points.  A
        conforming weak form never sees this term: there ``a`` is one weight
        inside an integral.
        """
        return 2.0 * self.contrast * points / self.radius**2

    # -- the exact spectrum ------------------------------------------------

    def exact_levels(self, count: int) -> list[tuple[float, int]]:
        """The lowest ``count`` distinct levels, each with its multiplicity."""
        return radial_levels(
            self.profile, orders=count + 2, per_order=max(4, count // 2)
        )[:count]

    def exact_ordered(self, count: int) -> np.ndarray:
        """The exact spectrum in order, repeated according to multiplicity.

        A Ritz solve returns the ordered spectrum, so this is what it has to be
        compared against; scoring against the distinct levels would record a
        correctly resolved threefold level as a large error.
        """
        values: list[float] = []
        for level, multiplicity in self.exact_levels(count + 4):
            values.extend([level] * multiplicity)
            if len(values) >= count:
                break
        return np.asarray(values[:count])

    def cluster_of(self, index: int, count: int) -> tuple[int, int]:
        """The ordered index range of the level occupying position ``index``.

        These are the indices :math:`k,\\ldots,k+m-1` of Section 2.2, at which
        Lemma 3.3 places the target eigenvalue cluster once the approximation of
        :math:`F_\\star` is below its threshold.
        """
        start = 0
        for _level, multiplicity in self.exact_levels(count + 4):
            if start <= index < start + multiplicity:
                return start, start + multiplicity
            start += multiplicity
        raise IndexError(index)

    def multiple_level_starts(self, count: int) -> list[int]:
        """Ordered indices at which a level of multiplicity above one begins."""
        starts, index = [], 0
        for _level, multiplicity in self.exact_levels(count + 4):
            if index + multiplicity > count:
                break
            if multiplicity > 1:
                starts.append(index)
            index += multiplicity
        return starts

    # -- what a method needs -----------------------------------------------

    def boundary_factor(self, points: np.ndarray) -> np.ndarray:
        r"""The factor :math:`\eta(x)=1-|x|^2` of Section 2.3.

        Smooth, positive inside, and vanishing exactly on the sphere, so it
        satisfies the conditions placed on :math:`\eta` there.
        """
        return 1.0 - np.sum(points**2, axis=1) / self.radius**2

    def boundary_factor_gradient(self, points: np.ndarray) -> np.ndarray:
        return -2.0 * points / self.radius**2

    def quadrature(self, level: int) -> tuple[np.ndarray, np.ndarray]:
        """Gauss in the radius, Gauss--Legendre in the polar cosine, uniform in
        the azimuth: ``level``, ``level+4`` and ``2*level+8`` points."""
        radial = level
        polar = level + 4
        azimuth = 2 * level + 8
        nodes, weights = np.polynomial.legendre.leggauss(radial)
        radii = 0.5 * self.radius * (nodes + 1.0)
        radial_weights = 0.5 * self.radius * weights * radii**2
        cosines, polar_weights = np.polynomial.legendre.leggauss(polar)
        sines = np.sqrt(1.0 - cosines**2)
        phis = 2.0 * np.pi * np.arange(azimuth) / azimuth
        phi_weight = 2.0 * np.pi / azimuth
        r_grid, c_index, p_index = np.meshgrid(
            np.arange(radial), np.arange(polar), np.arange(azimuth), indexing="ij"
        )
        r = radii[r_grid].ravel()
        cos_theta = cosines[c_index].ravel()
        sin_theta = sines[c_index].ravel()
        phi = phis[p_index].ravel()
        points = np.column_stack(
            [
                r * sin_theta * np.cos(phi),
                r * sin_theta * np.sin(phi),
                r * cos_theta,
            ]
        )
        weights_out = (
            radial_weights[r_grid].ravel()
            * polar_weights[c_index].ravel()
            * phi_weight
        )
        return points, weights_out

    def bounding_scale(self) -> float:
        """The length the sampled frequencies are non-dimensionalized by."""
        return self.radius

    def bounding_box(self):
        return np.full(3, -self.radius), np.full(3, self.radius)

    def contains(self, points: np.ndarray) -> np.ndarray:
        return np.sum(points**2, axis=1) < self.radius**2

    def structured_interior(self, per_axis: int) -> np.ndarray:
        """A uniform tensor grid of the bounding box, cut to the ball.

        The collocation baseline sums over its points instead of integrating,
        so how they are distributed decides how well that sum stands in for the
        integral.  A uniform grid is what its source prescribes.
        """
        lower, upper = self.bounding_box()
        axes = [
            lower[axis]
            + (np.arange(per_axis) + 0.5) * (upper[axis] - lower[axis]) / per_axis
            for axis in range(self.dimension)
        ]
        grids = np.meshgrid(*axes, indexing="ij")
        points = np.column_stack([grid.ravel() for grid in grids])
        return points[self.contains(points)]

    def sample_boundary(self, count: int, rng: np.random.Generator) -> np.ndarray:
        directions = rng.normal(size=(count, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        return self.radius * directions
