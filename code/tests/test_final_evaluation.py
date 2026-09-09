"""Independent checks of the final field integration and coordinate conventions."""

import numpy as np
import pytest

from rfmeig import features
from rfmeig.cube_product_integrals import cosine_product_pencil, fixed_cube_observables
from rfmeig.nonlinear import tensor_gauss_legendre_nd


@pytest.mark.parametrize("dimension,order", [(1, 100), (2, 48), (3, 24)])
@pytest.mark.parametrize("potential", ["square", "exp"])
def test_product_integrals_against_direct_gauss(dimension, order, potential):
    rng = np.random.default_rng(31415 + dimension)
    omega = rng.normal(0, 3, (7, dimension))
    phase = rng.uniform(0, 2 * np.pi, 7)
    points, weights = tensor_gauss_legendre_nd(order, ((0, 1),) * dimension)
    eta, gradient_eta = features.box_factor(points, normalize=True)
    angles = points @ omega.T + phase
    values = eta[:, None] * np.cos(angles)
    mass = values.T @ (weights[:, None] * values)
    energy = np.zeros_like(mass)
    for j in range(dimension):
        gradient = (
            gradient_eta[:, j, None] * np.cos(angles)
            - eta[:, None] * np.sin(angles) * omega[None, :, j]
        )
        energy += gradient.T @ (weights[:, None] * gradient)
    reaction = 20 * np.sum(points**2 if potential == "square" else np.exp(-np.pi * points), axis=1)
    energy += values.T @ ((weights * reaction)[:, None] * values)
    exact_energy, exact_mass = cosine_product_pencil(omega, phase, potential=potential)
    np.testing.assert_allclose(exact_mass, mass, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(exact_energy, energy, rtol=1e-11, atol=1e-12)


def test_constant_atom_on_ten_dimensional_cube():
    result = fixed_cube_observables(np.zeros((1, 10)), np.zeros(1), np.ones((1, 1)))
    # eta^2 induces ten Beta(3,3) factors; E[x^2]=2/7 and grad energy/mass=10d.
    np.testing.assert_allclose(result["mass"], [(8 / 15) ** 10], rtol=1e-13)
    np.testing.assert_allclose(result["rayleigh"], [100 + 200 * 2 / 7], rtol=1e-13)


def test_scalar_final_observables_are_scale_invariant():
    from rfmeig.box_basis import BoxFeatureBasis
    from rfmeig.experiments.exp5_gpe import BENCHMARK
    from rfmeig.final_evaluation import scalar_gpe_observables
    from rfmeig.problems.condensate import get_paper_benchmark

    problem = get_paper_benchmark(BENCHMARK)
    basis = BoxFeatureBasis.sample("rbf", 8, 0.2, 123, problem.bounds, center_spread=0.3)
    c = np.arange(1, 9, dtype=float) / 10
    a = scalar_gpe_observables(problem, basis, c, order=64)
    b = scalar_gpe_observables(problem, basis, -3 * c, order=64)
    np.testing.assert_allclose([a["lambda"], a["energy"]], [b["lambda"], b["energy"]], rtol=1e-13)


def test_dipolar_error_evaluation_uses_reference_nodes():
    from rfmeig.experiments.exp6_dipolar import make_basis
    from rfmeig.final_evaluation import dipolar_fields_on_reference_nodes
    from rfmeig.problems.dipolar import DipolarBenchmark, evaluate_feature_state_fields

    b = DipolarBenchmark()
    basis = make_basis(8, 123, b)
    c = (np.arange(8) + 1j * np.arange(8)[::-1], np.ones(8, dtype=complex))
    actual = dipolar_fields_on_reference_nodes(b, basis, c, (12, 16))
    # Deliberately rectangular: an accidental midpoint shift or transpose is visible.
    points = np.array(
        [
            (x, y)
            for x in np.linspace(-16, 16, 12, endpoint=False)
            for y in np.linspace(-16, 16, 16, endpoint=False)
        ]
    )
    expected = evaluate_feature_state_fields(basis, c, points)
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a.ravel(), e, atol=1e-13)
