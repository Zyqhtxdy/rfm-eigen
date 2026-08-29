"""Quadrature rules used to assemble the pencils.

The trial functions are smooth, so the integrands are smooth wherever the
coefficients are. Where a coefficient has a kink, the rule is made composite and
the kink is placed on a cell boundary, which restores the full Gauss order. The
Duffy transformation carries the tensor rule onto a simplex, and scrambled Sobol
points carry it into dimensions where a tensor rule is out of reach.

Which rule each example uses, and at what order, is listed in the
manuscript's appendix; every one of those entries is accompanied there by
the refinement that shows the reported error is the trial space's and not
the quadrature's.
"""

from __future__ import annotations

import numpy as np
from scipy.special import betaincinv, betaln


def gauss_legendre_unit(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss--Legendre nodes and weights on the unit interval."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def composite_gauss_unit(
    breakpoints: list[float], order: int
) -> tuple[np.ndarray, np.ndarray]:
    """A composite rule on ``[0, 1]`` with cells split at the breakpoints.

    A coefficient with a kink is only piecewise smooth, and a single Gauss rule
    across the kink loses its order. Splitting the interval there keeps every
    cell integrand smooth.
    """
    canonical_nodes, canonical_weights = np.polynomial.legendre.leggauss(order)
    endpoints = [0.0, *breakpoints, 1.0]
    nodes: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for left, right in zip(endpoints[:-1], endpoints[1:], strict=True):
        nodes.append(0.5 * (right - left) * canonical_nodes + 0.5 * (right + left))
        weights.append(0.5 * (right - left) * canonical_weights)
    return np.concatenate(nodes), np.concatenate(weights)


def tensor_square(
    order: int, breakpoints: list[float] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The tensor rule on the unit square, composite in each direction."""
    if breakpoints:
        nodes, weights = composite_gauss_unit(breakpoints, order)
    else:
        nodes, weights = gauss_legendre_unit(order)
    first, second = np.meshgrid(nodes, nodes, indexing="ij")
    first_weight, second_weight = np.meshgrid(weights, weights, indexing="ij")
    points = np.column_stack([first.ravel(), second.ravel()])
    return points, (first_weight * second_weight).ravel()


def tensor_cube(order: int, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """The tensor rule on the unit cube of the given dimension.

    The node count is ``order ** dimension``; beyond three or four dimensions a
    quasi-Monte Carlo rule is the only practical choice, see ``sobol_cube``.
    """
    nodes, weights = gauss_legendre_unit(order)
    grids = np.meshgrid(*([nodes] * dimension), indexing="ij")
    weight_grids = np.meshgrid(*([weights] * dimension), indexing="ij")
    points = np.column_stack([grid.ravel() for grid in grids])
    total = np.ones(points.shape[0])
    for grid in weight_grids:
        total *= grid.ravel()
    return points, total


def sobol_cube(
    count: int, dimension: int, seed: int, *, density: str = "uniform"
) -> tuple[np.ndarray, np.ndarray]:
    r"""Scrambled Sobol points on the unit cube, with the stated density.

    ``uniform`` is the plain rule.  ``beta22`` maps each coordinate through the
    inverse of the Beta(2, 2) distribution and divides by its density, which
    leaves every integral unchanged while placing more points where the boundary
    factor and the eigenfunctions carry their mass.  The choice is not cosmetic:
    at the same point count the two give visibly different accuracy, so it is
    recorded with every run and the uniform path is kept available as a check
    that the change of density has not changed what is being computed.

    The points come from the same generator the neural baseline of Example 3
    validates on.  That matters more than which generator it is: a comparison in
    which the two methods integrate on different nodes is a comparison of
    quadratures as much as of methods.
    """
    import torch

    engine = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=seed)
    sample = engine.draw(count).numpy().astype(np.float64)
    if density == "uniform":
        return sample, np.full(count, 1.0 / count)
    parameters = {"beta22": (2.0, 2.0), "beta33": (3.0, 3.0)}
    if density not in parameters:
        raise ValueError(f"unknown integration density: {density}")

    first, second = parameters[density]
    clipped = np.clip(sample, 2.0**-24, 1.0 - 2.0**-24)
    nodes = betaincinv(first, second, clipped)
    # The weight is the reciprocal density; taken in logarithms because it is a
    # product over ten coordinates and would otherwise underflow.
    log_density = np.sum(
        (first - 1.0) * np.log(nodes)
        + (second - 1.0) * np.log1p(-nodes)
        - betaln(first, second),
        axis=1,
    )
    return nodes, np.exp(-log_density) / count


def duffy_simplex(order: int, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """Tensor Gauss carried onto the ordered simplex by the Duffy transformation.

    The map ``x_1 = t_1``, ``x_2 = t_1 t_2``, ... sends the cube onto
    ``{0 < x_d < ... < x_1 < 1}``; its Jacobian is the product below. The
    transformation absorbs the vertex singularity of the simplex, so a tensor
    rule regains its order there.
    """
    nodes, weights = gauss_legendre_unit(order)
    grids = np.meshgrid(*([nodes] * dimension), indexing="ij")
    weight_grids = np.meshgrid(*([weights] * dimension), indexing="ij")
    cube = np.column_stack([grid.ravel() for grid in grids])
    total = np.ones(cube.shape[0])
    for grid in weight_grids:
        total *= grid.ravel()

    simplex = np.cumprod(cube, axis=1)
    jacobian = np.ones(cube.shape[0])
    for axis in range(dimension - 1):
        jacobian *= cube[:, axis] ** (dimension - 1 - axis)
    return simplex, total * jacobian
