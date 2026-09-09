r"""Example 3: the ten-dimensional Schrodinger operator on the unit cube.

The full-dimensional sampled pencil uses 2**22 scrambled Sobol points.
Separate one-dimensional reference solves provide the reference eigenvalues.
The first three values are reported, with the second and third belonging to
a repeated level. Product integration is used only to evaluate fixed RFM fields.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh_tridiagonal

from rfmeig import features

#: The dimension of the benchmark.
DIMENSION = 10
#: The strength of the potential.
POTENTIAL_SCALE = 20.0
#: The tail exponent of the frequency law; well above ``d/2 = 5``.
TAIL_EXPONENT = 8.0
#: The frequency scale.  It is not read off the operator -- the potential of this
#: problem has no wavevector to read -- so it is a blind choice, and is reported
#: as such.
FREQUENCY_SCALE = 2.0
#: Points used to assemble the RFM pencil; DRM training validation remains separate.
INTEGRATION_POINTS = 2**22
#: Base of the interior scramble seed; the potential's own seed is
#: ``base + 17 * len(name)``. Final evaluation uses a separate protocol.
SCRAMBLE_SEED_BASE = 203_406
#: Base of the boundary scramble seed.  Only the neural baseline needs it: its
#: iterates are not zero on the boundary, so its functional carries a boundary
#: term that has to be sampled.  The random feature trial space is zero there
#: by construction and has nothing to sample.
BOUNDARY_SCRAMBLE_SEED_BASE = 204_406
#: Points of the boundary sample, spread evenly over the twenty faces.
BOUNDARY_POINTS = 80_000


def potential(name: str, t: np.ndarray) -> np.ndarray:
    """One of the two one-dimensional potentials, without the strength factor."""
    if name == "square":
        return t * t
    if name == "exp":
        return np.exp(-math.pi * t)
    raise ValueError(f"unknown potential: {name}")


def one_dimensional_levels(
    name: str, scale: float, *, grid: int = 20_000, count: int = 4
) -> np.ndarray:
    r"""The lowest ``count`` levels of :math:`-u''+\gamma V u` on the unit interval.

    A second-order finite difference on a grid this fine resolves the low levels
    to far more digits than the experiment reports, and the matrix is
    tridiagonal, so the whole reference costs a fraction of a second.
    """
    step = 1.0 / (grid + 1)
    nodes = step * np.arange(1, grid + 1)
    diagonal = 2.0 / step**2 + scale * potential(name, nodes)
    off_diagonal = -np.ones(grid - 1) / step**2
    return eigh_tridiagonal(
        diagonal, off_diagonal, select="i", select_range=(0, count - 1)
    )[0]


def tensor_levels(levels: np.ndarray, dimension: int, count: int) -> list[float]:
    """The ``count`` smallest sums of ``dimension`` one-dimensional levels.

    Multiplicities are kept: the second and third entries of the list are equal
    for both potentials, and that is the point of the benchmark.  A heap is used
    because the full tensor grid has ``len(levels) ** dimension`` entries, of
    which only the first few are ever needed.
    """
    if count <= 0:
        return []
    if len(levels) < count:
        raise ValueError("not enough one-dimensional levels for the requested count")

    start = (0,) * dimension
    heap: list[tuple[float, tuple[int, ...]]] = [(float(dimension * levels[0]), start)]
    seen = {start}
    found: list[float] = []
    while heap and len(found) < count:
        value, index = heapq.heappop(heap)
        found.append(value)
        for axis in range(dimension):
            current = index[axis]
            if current + 1 >= len(levels):
                continue
            successor = list(index)
            successor[axis] = current + 1
            key = tuple(successor)
            if key in seen:
                continue
            seen.add(key)
            heapq.heappush(
                heap,
                (value - float(levels[current]) + float(levels[current + 1]), key),
            )
    if len(found) < count:
        raise RuntimeError("the heap ran out before the requested count was reached")
    return found


@dataclass(frozen=True)
class HighDimensionalCube:
    """The benchmark, its reference values, and the law its features are drawn from."""

    name: str = "square"
    dimension: int = DIMENSION
    scale: float = POTENTIAL_SCALE
    tail_exponent: float = TAIL_EXPONENT
    frequency_scale: float = FREQUENCY_SCALE

    def __post_init__(self) -> None:
        if self.name not in {"square", "exp"}:
            raise ValueError(f"unknown potential: {self.name}")

    # -- the operator -------------------------------------------------------
    def reaction(self, points: np.ndarray) -> np.ndarray:
        r""":math:`\gamma\sum_i V(x_i)`, sampled at the given points."""
        return self.scale * np.sum(potential(self.name, points), axis=1)

    def boundary_factor(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The product factor, normalized to one at the centre of the cube."""
        return features.box_factor(points, normalize=True)

    # -- the reference ------------------------------------------------------
    def reference_levels(self, count: int = 3, *, grid: int = 20_000) -> list[float]:
        """The lowest ``count`` exact eigenvalues, with multiplicity."""
        levels = one_dimensional_levels(
            self.name, self.scale, grid=grid, count=count + 1
        )
        return tensor_levels(levels, self.dimension, count)

    # -- the feature law ----------------------------------------------------
    @property
    def scramble_seed(self) -> int:
        """The scramble the interior rule uses, shared with the baseline."""
        return SCRAMBLE_SEED_BASE + 17 * len(self.name)

    @property
    def boundary_scramble_seed(self) -> int:
        """The scramble of the baseline's boundary sample; see above."""
        return BOUNDARY_SCRAMBLE_SEED_BASE + 17 * len(self.name)

    def draw_features(self, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        r"""``count`` features: one constant, the rest from the polynomial tail.

        The first feature is left at zero frequency.  It carries no information
        about the solution -- it is the boundary factor itself -- and it steadies
        the lowest mode, which is close to that shape.  The remainder are drawn
        from the isotropic beta-prime law at the stated scale.
        """
        rng = np.random.default_rng(seed)
        omega = np.zeros((count, self.dimension))
        phase = np.zeros(count)
        if count > 1:
            omega[1:], phase[1:] = features.sample_beta_prime_features(
                rng, count - 1, self.dimension, self.tail_exponent
            )
            omega[1:] *= self.frequency_scale
        return omega, phase
