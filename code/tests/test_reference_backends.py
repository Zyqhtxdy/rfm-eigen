from types import SimpleNamespace

import numpy as np
import pytest

from iaea_repro import reference
from rfmeig.baselines import sine_spectral_gpe
from rfmeig.baselines.gflm_ktm import GFLMKTMSettings, GFLMKTMSpectralSystem
from rfmeig.baselines.gflm_ktm_threaded import ThreadedGFLMKTMSystem
from rfmeig.experiments.exp4_iaea import run_neural
from rfmeig.problems.dipolar import DipolarBenchmark


def test_sine_reference_recovers_dirichlet_laplacian_ground_state():
    result = sine_spectral_gpe.solve(
        6, half_width=1., beta=0., potential=lambda x, y: np.zeros_like(x),
        max_iterations=400,
    )
    assert result.eigenvalue == pytest.approx(np.pi**2 / 2, rel=1e-12)
    assert result.energy == pytest.approx(result.eigenvalue / 2, rel=1e-12)
    assert result.stationarity_residual < 1e-10


def test_sine_reference_rejects_failed_linear_solve(monkeypatch):
    monkeypatch.setattr(sine_spectral_gpe, 'cg', lambda operator, rhs, **kwargs: (rhs, 1))
    with pytest.raises(RuntimeError, match='CG info=1'):
        sine_spectral_gpe.solve(4, max_iterations=1)


@pytest.mark.parametrize('workers', [1, 2])
def test_threaded_dipolar_transforms_preserve_discrete_operator(workers):
    benchmark = DipolarBenchmark()
    settings = GFLMKTMSettings(mesh_width=2., max_iterations=3)
    original = GFLMKTMSpectralSystem(benchmark, settings)
    threaded = ThreadedGFLMKTMSystem(benchmark, settings, workers=workers)
    states = (original.initial_state('gaussian', 0), original.initial_state('vortex', 1))
    for state in states:
        for actual, expected in zip(threaded.derivatives(state), original.derivatives(state), strict=True):
            np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-12)
        np.testing.assert_allclose(
            threaded.convolution(np.abs(state)**2), original.convolution(np.abs(state)**2),
            atol=1e-14, rtol=1e-12,
        )
    actual, expected = threaded.evaluate(states), original.evaluate(states)
    for result, baseline in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, baseline, atol=1e-12, rtol=1e-12)


def test_reactor_reference_is_immutable_and_checks_geometry(tmp_path):
    solution = reference.solve_reference(subdiv=1)
    path = tmp_path / 'reference.npz'
    reference.save_reference(solution, path)
    restored = reference.load_reference(path)
    assert restored.keff == solution.keff
    np.testing.assert_array_equal(restored.flux, solution.flux)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        reference.save_reference(solution, path)
    assert path.read_bytes() == before
    other = tmp_path / 'other_geometry.npz'
    np.savez(other, geometry_sha256=np.asarray('different geometry'))
    with pytest.raises(ValueError, match='geometry'):
        reference.load_reference(other)


def test_completed_neural_run_is_reused_before_training():
    row = {'method': 'drm', 'keff': 1.}
    calls = []

    def completed(call_id):
        calls.append(call_id)
        return row

    args = SimpleNamespace(neural_method='drm', neural_seed=17)
    assert run_neural(args, SimpleNamespace(completed=completed)) == [row]
    assert calls == ['neural_drm_17']
