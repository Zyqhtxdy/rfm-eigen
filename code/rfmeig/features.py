r"""Boundary factors and the laws the feature parameters are drawn from.

A random feature is a fixed function of a parameter drawn once and never
optimized.  Multiplying it by a factor that vanishes on the boundary places it
in :math:`H_0^1(\Omega)`, so the trial space is conforming for every
realization; that conformity is what allows Rayleigh--Ritz to be applied
pathwise, and it is the hypothesis the analysis of Section 3 rests on.

Two families of boundary factor appear in Section 4.  On a box the product
:math:`\prod_i x_i(1-x_i)` vanishes on every face.  On the ordered simplex no
product formula is available, and a determinant of sine modes is used instead:
it vanishes on the coordinate faces because a column vanishes, and on the
diagonal faces because two columns coincide.

Two sampling laws appear as well, and they are not interchangeable.  Experiment
1 draws the radius from the inverse cumulative distribution of a polynomial
tail; Experiments 2 and 3 draw it as the square root of a ratio of gamma
variates, which is the beta-prime law with the same tail exponent.  Each
experiment keeps the law its recorded numbers were produced with.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

# A boundary factor returns its values and its gradient at the given points.
BoundaryFactor = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


# --------------------------------------------------------------------------
# boundary factors
# --------------------------------------------------------------------------
def box_factor(
    points: np.ndarray, *, normalize: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    r""":math:`\eta(x)=\prod_i x_i(1-x_i)` on the unit box of any dimension.

    Smooth, positive inside and zero on every face.  Being an explicit smooth
    function rather than a distance, it lets the quotient of an eigenfunction by
    it inherit the regularity that the source condition of Section 3.4 asks for.

    Each partial derivative is the product of the other coordinates' factors,
    which is assembled from a running product from the left and one from the
    right rather than by dividing the whole product by one factor.  In ten
    dimensions the product is of size :math:`4^{-d}` and individual factors come
    arbitrarily close to zero near a face, so the division is the one step that
    would lose accuracy.  ``normalize`` multiplies by :math:`4^{d}`, which makes
    the factor one at the centre and keeps the entries of the pencil at a
    workable size; it cancels out of the Rayleigh quotient.
    """
    dimension = points.shape[1]
    factors = points * (1.0 - points)
    prefix = np.ones((points.shape[0], dimension + 1), dtype=points.dtype)
    suffix = np.ones_like(prefix)
    prefix[:, 1:] = np.cumprod(factors, axis=1)
    suffix[:, :-1] = np.cumprod(factors[:, ::-1], axis=1)[:, ::-1]

    scale = 4.0**dimension if normalize else 1.0
    values = scale * prefix[:, -1]
    gradients = scale * (1.0 - 2.0 * points) * prefix[:, :-1] * suffix[:, 1:]
    return values, gradients


def square_factor(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two-dimensional box factor, written out for speed."""
    x, y = points[:, 0], points[:, 1]
    eta_x, eta_y = x * (1.0 - x), y * (1.0 - y)
    gradients = np.column_stack([(1.0 - 2.0 * x) * eta_y, eta_x * (1.0 - 2.0 * y)])
    return eta_x * eta_y, gradients


def _sine_determinant(
    points: np.ndarray, orders: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""The determinant :math:`\det[\sin(k_i \pi x_j)]` and its gradient.

    The gradient is taken column by column: differentiating the determinant with
    respect to :math:`x_j` replaces column :math:`j` by its derivative and leaves
    the others alone.  Expanding this way rather than through the inverse keeps
    the formula valid where the matrix is singular, which is what happens on the
    faces.
    """
    count, dimension = points.shape
    values = np.empty((count, orders.size, dimension))
    derivatives = np.empty((count, orders.size, dimension))
    for axis in range(dimension):
        coordinate = points[:, axis]
        values[:, :, axis] = np.sin(np.pi * np.outer(coordinate, orders))
        derivatives[:, :, axis] = (
            np.pi * orders * np.cos(np.pi * np.outer(coordinate, orders))
        )
    determinant = np.linalg.det(values)
    gradient = np.empty((count, dimension))
    for axis in range(dimension):
        replaced = values.copy()
        replaced[:, :, axis] = derivatives[:, :, axis]
        gradient[:, axis] = np.linalg.det(replaced)
    return determinant, gradient


def simplex_factor(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r""":math:`\eta(x)=\det[\sin(k\pi x_j)]` on the ordered simplex, sign fixed.

    The orders are :math:`k=1,\dots,d`.  The sign is fixed by requiring the
    factor to be positive at an interior point, so that the same realization
    returns the same signed eigenfunctions on every run.
    """
    dimension = points.shape[1]
    orders = np.arange(1.0, dimension + 1.0)
    determinant, gradient = _sine_determinant(points, orders)
    interior = np.array(
        [[(dimension - index) / (dimension + 1.0) for index in range(dimension)]]
    )
    reference, _ = _sine_determinant(interior, orders)
    if reference[0] < 0.0:
        return -determinant, -gradient
    return determinant, gradient


def factor_scale(factor: BoundaryFactor, points: np.ndarray) -> float:
    """The largest value of a boundary factor, used to normalize it to one.

    The determinant factor is not normalized by construction, and its size
    changes with the dimension.  Dividing by this scale keeps the feature
    amplitudes comparable across experiments.
    """
    values, _ = factor(points)
    return float(np.max(np.abs(values)))


def scaled(factor: BoundaryFactor, scale: float) -> BoundaryFactor:
    """The same factor divided by a fixed scale."""

    def evaluate_scaled(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values, gradients = factor(points)
        return values / scale, gradients / scale

    return evaluate_scaled


# --------------------------------------------------------------------------
# sampling laws for the frequency
# --------------------------------------------------------------------------
def inverse_cdf_radius(
    rng: np.random.Generator, count: int, tail_exponent: float
) -> np.ndarray:
    r"""Radii from :math:`(1+r^2)^{-q}`, by inverting its cumulative distribution.

    Used by Example 1.  The exponent ``q`` controls how much mass sits at high
    frequency; Section 3.4 asks for :math:`d/2<q<s-1` when the quotient of the
    eigenfunction by the boundary factor lies in :math:`H^s`.
    """
    uniform = np.clip(rng.random(count), 1.0e-15, 1.0 - 1.0e-15)
    return np.sqrt((1.0 - uniform) ** (-1.0 / (tail_exponent - 1.0)) - 1.0)


def beta_prime_radius(
    rng: np.random.Generator, count: int, dimension: int, tail_exponent: float
) -> np.ndarray:
    r"""Radii from the beta-prime law with parameters :math:`(d/2, q-d/2)`.

    Used by Experiments 2 and 3.  Written as the square root of a ratio of gamma
    variates, which is exact and needs no inversion.  The resulting density on
    :math:`\mathbb{R}^d` decays like :math:`|\omega|^{-2q}`, the same tail as
    above, but the two laws differ at moderate radius and cannot be swapped
    without changing the recorded numbers.
    """
    shape = 0.5 * dimension
    numerator = rng.gamma(shape, 1.0, size=count)
    denominator = rng.gamma(tail_exponent - shape, 1.0, size=count)
    return np.sqrt(numerator / denominator)


def isotropic_directions(
    rng: np.random.Generator, count: int, dimension: int
) -> np.ndarray:
    """Uniform directions on the sphere, which make the frequency law isotropic.

    Normalized Gaussians, which is the construction that works in every
    dimension.  Example 1 draws its two-dimensional directions from an angle
    instead, and keeps that route inside its own sampler, because the two
    consume the generator differently and would not reproduce each other.
    """
    normal = rng.standard_normal((count, dimension))
    return normal / np.linalg.norm(normal, axis=1, keepdims=True)


def sample_inverse_cdf_features(
    count: int, tail_exponent: float, seed: int, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Example 1: inverted-tail frequency, phase uniform on a quarter turn.

    The draws are taken in this order -- radius, angle, phase -- from one
    generator, so that the seed alone reproduces the realization.
    """
    rng = np.random.default_rng(seed)
    radius = inverse_cdf_radius(rng, count, tail_exponent)
    angle = rng.uniform(0.0, 2.0 * math.pi, count)
    direction = np.column_stack([np.cos(angle), np.sin(angle)])
    omega = scale * radius[:, None] * direction
    phase = rng.uniform(0.0, 0.5 * math.pi, count)
    return omega, phase


def sample_beta_prime_features(
    rng: np.random.Generator, count: int, dimension: int, tail_exponent: float
) -> tuple[np.ndarray, np.ndarray]:
    r"""Experiments 2 and 3: beta-prime frequency, phase in :math:`\{0,\pi/2\}`.

    The two-point phase gives each direction both a cosine and a sine, which is
    what a real trigonometric feature needs in order to represent a shift.
    """
    direction = isotropic_directions(rng, count, dimension)
    radius = beta_prime_radius(rng, count, dimension, tail_exponent)
    phase = rng.choice([0.0, 0.5 * math.pi], size=count)
    return direction * radius[:, None], phase


@dataclass(frozen=True)
class CarrierMode:
    r"""One frequency the coefficients of the operator put into the solution.

    A coefficient that oscillates like :math:`\cos(a\cdot x)` forces the
    eigenfunction to carry the frequency :math:`a` and its harmonics, whatever
    the boundary condition does.  Drawing part of the features near those
    frequencies rather than from the tail alone is the only problem-dependent
    choice in the method, and it is stated in the manuscript as such.  The
    ``weight`` is the probability the mode receives, before normalization.
    """

    weight: float
    frequency: np.ndarray
    phase: float = 0.0

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self.frequency))

    @property
    def direction(self) -> np.ndarray:
        return np.asarray(self.frequency, dtype=float) / self.magnitude


def harmonic_carrier_modes(
    wavevectors: Sequence[np.ndarray], *, harmonics: int
) -> tuple[CarrierMode, ...]:
    r"""Every harmonic of every wavevector, with both phases and equal weight.

    Taking harmonics up to a fixed order rather than the wavevectors alone is
    what covers the frequencies a product of two oscillating coefficients
    generates.  Both phases appear so that the pair spans a shift.
    """
    modes: list[CarrierMode] = []
    for base in wavevectors:
        base = np.asarray(base, dtype=float)
        for harmonic in range(1, harmonics + 1):
            frequency = harmonic * base
            for phase in (0.0, 0.5 * math.pi):
                modes.append(CarrierMode(1.0, frequency, phase))
    return tuple(modes)


def sample_operator_informed_features(
    rng: np.random.Generator,
    count: int,
    dimension: int,
    tail_exponent: float,
    modes: Sequence[CarrierMode],
    *,
    atom_probability: float,
    tail_probability: float,
    carrier_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""A three-part mixture: a zero atom, a heavy tail, and the carrier modes.

    Each feature is assigned to one part by a single uniform draw, so that
    changing a probability changes which features are drawn but not the number
    of them.  The atom leaves :math:`\omega=0`, which contributes the boundary
    factor itself and costs nothing.  The tail supplies the frequencies the
    source condition of Section 3.4 needs.  The remainder is placed near a
    carrier mode, with a Gaussian spread of width ``carrier_width`` so that the
    features cover a neighbourhood of the mode rather than the mode alone.
    """
    if atom_probability + tail_probability >= 1.0:
        raise ValueError("the atom and tail probabilities must leave room for carriers")

    omega = np.zeros((count, dimension))
    phase = np.zeros(count)
    assignment = rng.random(count)

    tail = (assignment >= atom_probability) & (
        assignment < atom_probability + tail_probability
    )
    tail_count = int(tail.sum())
    if tail_count:
        omega[tail], phase[tail] = sample_beta_prime_features(
            rng, tail_count, dimension, tail_exponent
        )

    carriers = np.flatnonzero(assignment >= atom_probability + tail_probability)
    if carriers.size:
        weights = np.asarray([mode.weight for mode in modes], dtype=float)
        weights /= weights.sum()
        if np.all(weights == weights[0]):
            chosen = rng.integers(0, len(modes), size=carriers.size)
        else:
            chosen = rng.choice(len(modes), size=carriers.size, p=weights)
        signs = rng.choice([-1.0, 1.0], size=carriers.size)
        for local, index in enumerate(carriers):
            mode = modes[chosen[local]]
            omega[index] = signs[local] * mode.magnitude * mode.direction
            omega[index] += rng.normal(scale=carrier_width, size=dimension)
            phase[index] = mode.phase
    return omega, phase


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def evaluate(
    points: np.ndarray,
    omega: np.ndarray,
    phase: np.ndarray,
    factor: BoundaryFactor,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Values and gradients of :math:`\eta(x)\cos(\omega\cdot x+b)` at every point.

    Returns arrays of shape ``(points, features)`` and
    ``(points, features, dimension)``, the second being the product rule applied
    to the boundary factor and the cosine.
    """
    eta, grad_eta = factor(points)
    argument = points @ omega.T + phase[None, :]
    cosine = np.cos(argument)
    sine = np.sin(argument)
    values = eta[:, None] * cosine
    gradients = (
        grad_eta[:, None, :] * cosine[:, :, None]
        - eta[:, None, None] * sine[:, :, None] * omega[None, :, :]
    )
    return values, gradients
