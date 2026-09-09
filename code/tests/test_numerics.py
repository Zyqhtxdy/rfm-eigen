import numpy as np
import pandas as pd
import pytest
from scipy.linalg import eigh

from paper_assets.paths import REPO_DATA
from rfmeig import features, metrics, quadrature
from rfmeig.experiments import exp1_interior_double as exp1
from rfmeig.problems.separable import (
    DoubleKinkPotential,
    leading_pairs,
    locate_pair,
    product_fields,
    separable_reaction,
    solve_one_dimensional,
)
from rfmeig.rayleigh_ritz import (
    assemble_cosine_pencil,
    assemble_cosine_pencil_expanded,
    assemble_pencil,
    solve_energy_whitened,
    solve_mass_whitened,
)
from rfmeig.separable_assembly import SeparableSquareQuadrature


def test_constant_diffusion_matches_pointwise_coefficient():
    points, weights = quadrature.tensor_square(12)
    omega, phase = features.sample_inverse_cdf_features(16, 2.42, 91, 3.)
    values, gradients = features.evaluate(points, omega, phase, features.square_factor)
    scalar = assemble_pencil(values, gradients, weights, diffusion=2., reaction=3.)
    sampled = assemble_pencil(values, gradients, weights, diffusion=np.full(len(points), 2.), reaction=3.)
    for actual, expected in zip(scalar, sampled, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)
    blocked = assemble_cosine_pencil(points, weights, omega, phase, features.square_factor,
                                    diffusion=2., reaction=3., block=29)
    for actual, expected in zip(blocked, scalar, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-12)


def test_expanded_assembly_matches_direct_gradient_integrals():
    points, weights = quadrature.tensor_square(15)
    omega, phase = features.sample_inverse_cdf_features(20, 2.42, 41, 2.)
    values, gradients = features.evaluate(points, omega, phase, features.square_factor)
    direct = assemble_pencil(values, gradients, weights, reaction=4.)
    expanded = assemble_cosine_pencil_expanded(points, weights, omega, phase, features.square_factor,
                                              reaction=4., block=37)
    for actual, expected in zip(expanded, direct, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-12)


@pytest.mark.parametrize("solver", [solve_energy_whitened, solve_mass_whitened])
def test_whitening_preserves_eigenvalues_and_mass_normalization(solver):
    rng = np.random.default_rng(713)
    change = rng.normal(size=(8, 8)) + 4*np.eye(8)
    mass = change.T @ change
    energy = change.T @ np.diag(np.arange(1., 9.)) @ change
    result = solver(energy, mass, 5, tolerance=1e-12)
    np.testing.assert_allclose(result.eigenvalues, eigh(energy, mass, eigvals_only=True)[:5], rtol=1e-12)
    np.testing.assert_allclose(result.coefficients.T @ mass @ result.coefficients, np.eye(5), atol=1e-12)


@pytest.mark.parametrize("bad", [0., -1., np.nan, np.inf])
@pytest.mark.parametrize("solver", [solve_energy_whitened, solve_mass_whitened])
def test_invalid_truncation_is_rejected(solver, bad):
    with pytest.raises(ValueError):
        solver(np.eye(3), np.eye(3), 1, tolerance=bad)


def test_separated_rule_matches_direct_assembly_and_h1_projection():
    potential = DoubleKinkPotential()
    vals, vectors = solve_one_dimensional(potential, modes=24, order=32)
    pairs = leading_pairs(vals, 8, limit=18) + [(1, 2), (2, 1)]
    omega, phase = features.sample_inverse_cdf_features(64, 2.42, 123, 5.)
    rule = SeparableSquareQuadrature(potential, vectors, pairs, 32)
    forms = rule.assemble(omega, phase)
    points, weights = quadrature.tensor_square(32, potential.breakpoints)
    phi, gradients = features.evaluate(points, omega, phase, features.square_factor)
    energy, mass = assemble_pencil(phi, gradients, weights, reaction=separable_reaction(points, potential))
    np.testing.assert_allclose(forms.energy, energy, atol=1e-13, rtol=1e-12)
    np.testing.assert_allclose(forms.mass, mass, atol=1e-15, rtol=1e-12)
    exact = product_fields(points, vectors, pairs)
    np.testing.assert_allclose(forms.cross, metrics.h1_gram(weights, *exact, phi, gradients), atol=1e-12)
    solution = solve_energy_whitened(energy, mass, 10, tolerance=1e-12)
    basis = solution.basis
    direct = metrics.best_approximation_error(
        weights, exact[0][:, :8], exact[1][:, :8], phi @ basis,
        np.einsum('qnd,nj->qjd', gradients, basis, optimize=True), relative_tolerance=1e-13,
    )
    separated = metrics.best_approximation_from_grams(
        forms.reference_gram[:8, :8], basis.T @ forms.h1 @ basis, forms.cross[:8] @ basis,
        relative_tolerance=1e-13,
    )
    assert separated == pytest.approx(direct, rel=1e-5)


@pytest.mark.parametrize("n,sample", [(128, 0), (216, 85), (896, 56)])
def test_refined_driver_reproduces_recorded_errors(n, sample):
    potential = DoubleKinkPotential()
    values, vectors = solve_one_dimensional(potential, modes=112, order=96)
    first, last, _ = locate_pair(values, (1, 2), limit=18)
    pairs = leading_pairs(values, last, limit=18) + [(1, 2), (2, 1)]
    rule = SeparableSquareQuadrature(potential, vectors, pairs, 512)
    result = exp1.run_draw(
        feature_count=n, seed=exp1.draw_seed(exp1.DEFAULT_BASE_SEED, n, sample),
        potential=potential, tail_exponent=2.42, frequency_scale=8.5, truncation=1e-12,
        assembly=rule, evaluation=rule, subspace=rule, first=first, last=last,
        exact=exp1.DEFAULT_REFERENCE_VALUE,
    )
    data = pd.read_csv(REPO_DATA / 'experiment1_theory_matched_trials.csv', float_precision='round_trip')
    recorded = data[(data.N == n) & (data['sample'] == sample)].iloc[0]
    for field in exp1.RATE_FIELDS:
        assert result[field] == pytest.approx(recorded[field], rel=5e-5)
    assert result['retained'] == recorded['rank']


def test_mean_bootstrap_uses_draws_and_all_budgets():
    rows = [{'N': n, 'error': amplitude/n} for n in [16, 32, 64, 128]
            for amplitude in [1., 1., 1., 1.]]
    low, high = metrics.bootstrap_slope_interval(rows, [16, 32, 64, 128], 'error',
                                                statistic='mean', replicates=100)
    assert low == pytest.approx(-1.)
    assert high == pytest.approx(-1.)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_torch_assembly_matches_numpy(device):
    torch = pytest.importorskip('torch')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    from rfmeig import gpu
    points, weights = quadrature.tensor_square(12)
    omega, phase = features.sample_inverse_cdf_features(16, 2.42, 77, 2.)
    reference = assemble_cosine_pencil_expanded(
        points, weights, omega, phase, features.square_factor, reaction=3., block=29,
    )
    actual = gpu.cosine_pencil_expanded(
        points, weights, omega, phase, features.square_factor, reaction=3., block=29, device=device,
    )
    for result, expected in zip(actual, reference, strict=True):
        np.testing.assert_allclose(result.cpu().numpy(), expected, atol=1e-13, rtol=1e-11)
