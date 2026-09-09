"""GFLM--KTM with SciPy's configurable FFT worker budget.

The grid, truncated kernel, time step, stabilization, stopping rule and initial
states are inherited from the NumPy-transform implementation. The reference
backend tests compare derivatives, convolution, energy and chemical potentials
between the two implementations to floating-point accuracy.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
import scipy.fft

from rfmeig.baselines.gflm_ktm import (
    ComplexArray,
    GFLMKTMResult,
    GFLMKTMSettings,
    GFLMKTMSpectralSystem,
    HistoryCallback,
    HistoryRecord,
)
from rfmeig.problems.dipolar import DipolarBenchmark


class ThreadedGFLMKTMSystem(GFLMKTMSpectralSystem):
    """The published system with scipy's threaded transform substituted."""

    def __init__(self, benchmark: DipolarBenchmark,
                 settings: GFLMKTMSettings, workers: int = 1) -> None:
        super().__init__(benchmark, settings)
        self.workers = int(workers)

    def _fft2(self, array):
        return scipy.fft.fft2(array, workers=self.workers)

    def _ifft2(self, array):
        return scipy.fft.ifft2(array, workers=self.workers)

    def convolution(self, density):
        padded = np.zeros(self.dipolar_symbol.shape)
        padded[self.padding_slice] = density
        potential = self._ifft2(self.dipolar_symbol * self._fft2(padded)).real
        return potential[self.padding_slice]

    def derivatives(self, state):
        transformed = self._fft2(state)
        derivative_x = self._ifft2(1j * self.kx * transformed)
        derivative_y = self._ifft2(1j * self.ky * transformed)
        negative_laplacian = self._ifft2(self.wave_number_squared * transformed)
        return derivative_x, derivative_y, negative_laplacian


def solve_gflm_ktm_threaded(
    benchmark: DipolarBenchmark,
    settings: GFLMKTMSettings,
    initial_kinds: tuple[str, str] = ("gaussian", "gaussian"),
    workers: int = 1,
    *,
    callback: HistoryCallback | None = None,
    capture_history: bool = False,
    system: ThreadedGFLMKTMSystem | None = None,
) -> GFLMKTMResult:
    """The published fixed-point loop, verbatim, on the threaded system."""
    started = perf_counter()
    if system is None:
        system = ThreadedGFLMKTMSystem(
            benchmark, settings, workers=workers
        )
    elif system.settings != settings or system.benchmark != benchmark:
        raise ValueError("the supplied spectral system does not match the run")
    states: tuple[ComplexArray, ComplexArray] = tuple(
        system.initial_state(kind, component)
        for component, kind in enumerate(initial_kinds)
    )
    history: list[HistoryRecord] | None = [] if capture_history else None

    def emit(record: HistoryRecord) -> None:
        snapshot = dict(record)
        if history is not None:
            history.append(snapshot)
        if callback is not None:
            callback(dict(snapshot))

    fixed_point_residual = np.inf
    energy, chemical_potentials, virial, dipolar_potentials = system.evaluate(
        states
    )
    mass_errors = tuple(
        abs(system.weight * float(np.sum(np.abs(state) ** 2)) - mass)
        for state, mass in zip(states, benchmark.masses)
    )
    emit({
        "event": "initial",
        "iteration": 0,
        "initial_kind_1": initial_kinds[0],
        "initial_kind_2": initial_kinds[1],
        "cumulative_seconds": perf_counter() - started,
        "energy": energy,
        "mu1": chemical_potentials[0],
        "mu2": chemical_potentials[1],
        "fixed_point_residual": fixed_point_residual,
        "virial_residual": abs(virial),
        "constraint_error_1": mass_errors[0],
        "constraint_error_2": mass_errors[1],
        "step_length": settings.time_step,
        "accepted_step_norm": 0.0,
        "optimizer_status": "iterating",
    })
    accepted_iterations = 0

    for _ in range(settings.max_iterations):
        densities = tuple(np.abs(state) ** 2 for state in states)
        beta = np.asarray(benchmark.beta)
        dipolar = np.asarray(benchmark.dipolar)
        next_states = []
        differences = []
        for component, state in enumerate(states):
            effective = sum(
                beta[component, other] * densities[other]
                + dipolar[component, other] * dipolar_potentials[other]
                for other in range(2)
            )
            total_potential = system.potential + effective
            stabilization = 0.5 * (float(np.max(total_potential))
                                   + float(np.min(total_potential)))
            angular = system.angular_momentum(state)
            right_hand_side = (
                state / settings.time_step
                + (stabilization - total_potential
                   + chemical_potentials[component]) * state
                + benchmark.rotation * angular
            )
            transformed = system._fft2(right_hand_side)
            candidate = system._ifft2(
                transformed
                / (1.0 / settings.time_step + stabilization
                   + 0.5 * system.wave_number_squared)
            )
            candidate = system.normalize(candidate, benchmark.masses[component])
            differences.append(float(np.max(np.abs(candidate - state))))
            next_states.append(candidate)
        states = (next_states[0], next_states[1])
        accepted_iterations += 1
        fixed_point_residual = max(differences)
        energy, chemical_potentials, virial, dipolar_potentials = system.evaluate(
            states
        )
        mass_errors = tuple(
            abs(system.weight * float(np.sum(np.abs(state) ** 2)) - mass)
            for state, mass in zip(states, benchmark.masses)
        )
        emit({
            "event": "iteration",
            "iteration": accepted_iterations,
            "initial_kind_1": initial_kinds[0],
            "initial_kind_2": initial_kinds[1],
            "cumulative_seconds": perf_counter() - started,
            "energy": energy,
            "mu1": chemical_potentials[0],
            "mu2": chemical_potentials[1],
            "fixed_point_residual": fixed_point_residual,
            "virial_residual": abs(virial),
            "constraint_error_1": mass_errors[0],
            "constraint_error_2": mass_errors[1],
            "step_length": settings.time_step,
            "accepted_step_norm": fixed_point_residual,
            "optimizer_status": (
                "converged" if fixed_point_residual <= settings.tolerance
                else "iterating"
            ),
        })
        if fixed_point_residual <= settings.tolerance:
            break

    converged = fixed_point_residual <= settings.tolerance
    optimizer_status = "converged" if converged else "maximum_iterations"
    optimizer_message = (
        "fixed-point residual reached tolerance" if converged
        else "maximum iteration count reached before residual tolerance"
    )
    emit({
        "event": "final",
        "iteration": accepted_iterations,
        "initial_kind_1": initial_kinds[0],
        "initial_kind_2": initial_kinds[1],
        "cumulative_seconds": perf_counter() - started,
        "energy": energy,
        "mu1": chemical_potentials[0],
        "mu2": chemical_potentials[1],
        "fixed_point_residual": fixed_point_residual,
        "virial_residual": abs(virial),
        "constraint_error_1": mass_errors[0],
        "constraint_error_2": mass_errors[1],
        "step_length": 0.0,
        "accepted_step_norm": 0.0,
        "optimizer_status": optimizer_status,
    })
    return GFLMKTMResult(
        energy=energy,
        chemical_potentials=chemical_potentials,
        virial_residual=abs(virial),
        fixed_point_residual=fixed_point_residual,
        iterations=accepted_iterations,
        converged=converged,
        grid_size=system.size,
        degrees_of_freedom=2 * system.size * system.size,
        wall_seconds=perf_counter() - started,
        states=states,
        initial_kinds=initial_kinds,
        optimizer_success=converged,
        optimizer_status=optimizer_status,
        optimizer_message=optimizer_message,
        history=history,
    )
