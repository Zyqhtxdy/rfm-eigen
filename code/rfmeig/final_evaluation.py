"""Evaluate fixed numerical states independently of their optimization rules."""

from __future__ import annotations

import numpy as np

from rfmeig.nonlinear import tensor_gauss_legendre_nd


def scalar_gpe_observables(benchmark, basis, coefficients, *, order=112, block=2048):
    """Mass-normalized scalar GPE observables on a final Gauss rule.

    The coefficients and the optimizer's stopping residual are unchanged.
    The energy uses the scalar benchmark convention beta * integral(u**4) / 4.
    """
    points, weights = tensor_gauss_legendre_nd(order, benchmark.bounds)
    mass = linear = quartic = 0.0
    for start in range(0, len(points), block):
        x = points[start : start + block]
        w = weights[start : start + block]
        values, gradients, _ = basis.evaluate(x)
        u = values @ coefficients
        mass += float(w @ (u * u))
        linear += float(
            w @ (sum((g @ coefficients) ** 2 for g in gradients) + benchmark.potential(x) * u * u)
        )
        quartic += float(w @ (u**4))
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("the fixed state must have positive finite mass")
    return {
        "evaluation_mass": mass,
        "evaluation_order": int(order),
        "lambda": linear / mass + benchmark.beta * quartic / mass**2,
        "energy": 0.5 * linear / mass + 0.25 * benchmark.beta * quartic / mass**2,
    }


def dipolar_fields_on_reference_nodes(benchmark, basis, coefficients, shape):
    """Evaluate on the left-endpoint periodic nodes used by GFLM--KTM.

    Energy integration remains on its existing midpoint rule. No interpolation
    or periodic shift of a stored midpoint array is used for the field error.
    """
    from rfmeig.problems.dipolar import evaluate_feature_state_fields

    if len(shape) != 2 or min(shape) < 2:
        raise ValueError("a two-dimensional reference grid is required")
    axes = [
        lower + np.arange(size) * (upper - lower) / size
        for (lower, upper), size in zip(benchmark.bounds, shape, strict=True)
    ]
    grids = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([grid.ravel() for grid in grids])
    return tuple(
        field.reshape(shape) for field in evaluate_feature_state_fields(basis, coefficients, points)
    )
