r"""The conforming sampled trial space on the ball, assembled and solved.

This is the construction of Section 2.3 on the geometry of Example 2: every
trial function is a sampled cosine multiplied by the boundary factor,

.. math::
    \phi_\theta(x)=\eta(x)\chi_\theta(x),
    \qquad \chi_\theta(x)=\cos(\omega\cdot x+b),
    \qquad \eta(x)=1-|x|^2 ,

so the space sits inside :math:`H_0^1(\Omega)` exactly, for every realization,
and the discrete problem is the symmetric generalized eigenproblem
:math:`K\alpha=\lambda M\alpha` of Section 2.3 in a conforming subspace.  Only
the frequency and the phase are sampled; nothing is trained.

The diffusion coefficient enters the stiffness integrand as a weight,
:math:`K_{pq}=\int_\Omega a\nabla\phi_p\cdot\nabla\phi_q`, and nowhere else.
That is the whole cost of a variable coefficient to a conforming weak form,
and it is worth contrasting with what the same coefficient does to the strong
residual of the collocation baseline, where it adds a first-order term.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from rfmeig.rayleigh_ritz import solve_mass_whitened


@dataclass
class Result:
    values: np.ndarray
    dimension: int
    retained: int
    assemble_seconds: float
    solve_seconds: float
    seconds: float


def sample_frequencies(
    count: int, dimension: int, scale: float, rng: np.random.Generator
) -> np.ndarray:
    r"""Isotropic frequencies: a uniform direction times a sampled magnitude.

    The direction carries no preference, which is what keeps a symmetric
    domain's multiple levels from being split by the sampling itself.  The
    magnitude is :math:`\sqrt{X/Y}` with :math:`X\sim\Gamma(5,1)` and
    :math:`Y\sim\Gamma(3,1)`.

    :func:`rfmeig.features.sample_beta_prime_features` is the same shape of law
    but with the parameters tied to the dimension and the tail exponent, which
    are not these; it is deliberately not reused here, because substituting it
    would change every recorded number of Example 2 without changing the
    method.
    """
    directions = rng.normal(size=(count, dimension))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    magnitude = np.sqrt(rng.gamma(5.0, 1.0, count) / rng.gamma(3.0, 1.0, count))
    return scale * magnitude[:, None] * directions


def assemble(
    domain, features: int, seed: int, scale: float, quadrature_level: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The stiffness and mass matrices of the sampled space, on one draw.

    Returns ``(stiffness, mass, points, weights)``.  Both matrices are formed
    as Gram matrices of quadrature-weighted values, which keeps them symmetric
    positive semidefinite by construction rather than by symmetrization after
    the fact.
    """
    rng = np.random.default_rng(seed)
    points, weights = domain.quadrature(quadrature_level)

    frequencies = sample_frequencies(
        features, domain.dimension, scale / domain.bounding_scale(), rng
    )
    phases = rng.choice([0.0, 0.5 * np.pi], size=features)

    factor = domain.boundary_factor(points)
    factor_gradient = domain.boundary_factor_gradient(points)
    angles = points @ frequencies.T + phases
    cosines = np.cos(angles)
    sines = np.sin(angles)

    root = np.sqrt(weights)[:, None]
    scaled_values = root * (factor[:, None] * cosines)
    mass = scaled_values.T @ scaled_values

    # the product rule for grad(eta * cos(omega.x + b)), with the diffusion
    # coefficient carried as sqrt(a) on each factor of the Gram product
    conductivity = np.sqrt(domain.coefficient(points))[:, None]
    stiffness = np.zeros((features, features))
    for axis in range(domain.dimension):
        component = (
            factor_gradient[:, axis][:, None] * cosines
            - factor[:, None] * sines * frequencies[:, axis][None, :]
        )
        scaled = root * conductivity * component
        stiffness += scaled.T @ scaled
    return stiffness, mass, points, weights


def solve(
    domain,
    features: int,
    seed: int,
    scale: float,
    quadrature_level: int,
    count: int,
    truncation: float = 1.0e-12,
) -> Result:
    """Assemble on the domain's rule and take the lowest ``count`` levels.

    The clock covers assembly and solution together, which is what the times
    of Table 2 report; the error evaluation lies outside it.
    """
    started = time.perf_counter()
    stiffness, mass, _points, _weights = assemble(
        domain, features, seed, scale, quadrature_level
    )
    assembled = time.perf_counter()

    solution = solve_mass_whitened(stiffness, mass, count, tolerance=truncation)
    finished = time.perf_counter()
    return Result(
        values=np.asarray(solution.eigenvalues),
        dimension=features,
        retained=int(solution.retained),
        assemble_seconds=assembled - started,
        solve_seconds=finished - assembled,
        seconds=finished - started,
    )
