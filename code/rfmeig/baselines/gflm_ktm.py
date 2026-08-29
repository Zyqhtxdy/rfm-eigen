r"""The gradient flow baseline of Example 6.

The method of Li et al.: a normalized gradient flow discretized by a
Krylov-accelerated time-marching scheme on a uniform mesh.  It descends the same
energy the random feature method minimizes, so the comparison between them is
between how each represents the state and not between what each is computing.

Its wall time depends strongly on where it starts -- across the sampled initial
pairs it varies by nearly an order of magnitude -- which is why the reported time
is a median over a fixed set of initial pairs rather than the time of one run.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from rfmeig.problems.dipolar import DipolarBenchmark
from rfmeig.problems.dipolar_kernels import truncated_dipolar_symbol_2d

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]
HistoryRecord: TypeAlias = dict[str, float | int | bool | str]
HistoryCallback: TypeAlias = Callable[[HistoryRecord], None]

PUBLISHED_INITIAL_KINDS = (
    "gaussian",
    "vortex",
    "conjugate_vortex",
    "mixed",
    "conjugate_mixed",
    "rotation_mixed",
    "conjugate_rotation_mixed",
    "counter_rotation_mixed",
    "conjugate_counter_rotation_mixed",
    "thomas_fermi",
)


@dataclass(frozen=True)
class GFLMKTMSettings:
    mesh_width: float
    time_step: float = 0.1
    tolerance: float = 1e-11
    max_iterations: int = 100000
    padding_factor: int = 3


@dataclass
class GFLMKTMResult:
    energy: float
    chemical_potentials: tuple[float, float]
    virial_residual: float
    fixed_point_residual: float
    iterations: int
    converged: bool
    grid_size: int
    degrees_of_freedom: int
    wall_seconds: float
    states: tuple[ComplexArray, ComplexArray]
    initial_kinds: tuple[str, str] = ("gaussian", "gaussian")
    optimizer_success: bool = False
    optimizer_status: str = "not_started"
    optimizer_message: str = ""
    history: list[HistoryRecord] | None = None


class GFLMKTMSpectralSystem:
    def __init__(
        self,
        benchmark: DipolarBenchmark,
        settings: GFLMKTMSettings,
    ) -> None:
        self.benchmark = benchmark
        self.settings = settings
        lower, upper = benchmark.bounds[0]
        requested = (upper - lower) / settings.mesh_width
        self.size = int(round(requested))
        if not np.isclose(requested, self.size):
            raise ValueError("The mesh width must divide the box length")
        self.spacing = (upper - lower) / self.size
        nodes = lower + self.spacing * np.arange(self.size)
        self.x, self.y = np.meshgrid(nodes, nodes, indexing="ij")
        self.weight = self.spacing**2
        self.potential = 0.5 * (self.x * self.x + self.y * self.y)
        frequencies = 2.0 * np.pi * np.fft.fftfreq(
            self.size, d=self.spacing
        )
        self.kx, self.ky = np.meshgrid(
            frequencies, frequencies, indexing="ij"
        )
        self.wave_number_squared = self.kx * self.kx + self.ky * self.ky

        padded_size = settings.padding_factor * self.size
        padded_frequencies = 2.0 * np.pi * np.fft.fftfreq(
            padded_size, d=self.spacing
        )
        padded_kx, padded_ky = np.meshgrid(
            padded_frequencies, padded_frequencies, indexing="ij"
        )
        half_width = 0.5 * (upper - lower)
        self.dipolar_symbol = truncated_dipolar_symbol_2d(
            padded_kx,
            padded_ky,
            benchmark.dipole_axis,
            2.0 * np.sqrt(2.0) * half_width,
        )
        offset = (padded_size - self.size) // 2
        self.padding_slice = (
            slice(offset, offset + self.size),
            slice(offset, offset + self.size),
        )

    def convolution(self, density: RealArray) -> RealArray:
        padded = np.zeros(self.dipolar_symbol.shape)
        padded[self.padding_slice] = density
        potential = np.fft.ifft2(
            self.dipolar_symbol * np.fft.fft2(padded)
        ).real
        return potential[self.padding_slice]

    def derivatives(
        self, state: ComplexArray
    ) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
        transformed = np.fft.fft2(state)
        derivative_x = np.fft.ifft2(1j * self.kx * transformed)
        derivative_y = np.fft.ifft2(1j * self.ky * transformed)
        negative_laplacian = np.fft.ifft2(
            self.wave_number_squared * transformed
        )
        return derivative_x, derivative_y, negative_laplacian

    def angular_momentum(self, state: ComplexArray) -> ComplexArray:
        derivative_x, derivative_y, _ = self.derivatives(state)
        return -1j * (self.x * derivative_y - self.y * derivative_x)

    def normalize(
        self, state: ComplexArray, target_mass: float
    ) -> ComplexArray:
        norm = np.sqrt(self.weight * np.sum(np.abs(state) ** 2))
        return np.sqrt(target_mass) * state / norm

    def evaluate(
        self, states: tuple[ComplexArray, ComplexArray]
    ) -> tuple[
        float,
        tuple[float, float],
        float,
        tuple[RealArray, RealArray],
    ]:
        densities = tuple(np.abs(state) ** 2 for state in states)
        dipolar_potentials = tuple(
            self.convolution(density) for density in densities
        )
        beta = np.asarray(self.benchmark.beta)
        dipolar = np.asarray(self.benchmark.dipolar)
        kinetic = 0.0
        potential_energy = 0.0
        rotation = 0.0
        pde_actions = []
        interaction = 0.0
        dipolar_energy = 0.0
        for component, state in enumerate(states):
            derivative_x, derivative_y, negative_laplacian = (
                self.derivatives(state)
            )
            angular = -1j * (
                self.x * derivative_y - self.y * derivative_x
            )
            kinetic += 0.5 * self.weight * float(
                np.sum(
                    np.abs(derivative_x) ** 2
                    + np.abs(derivative_y) ** 2
                )
            )
            potential_energy += self.weight * float(
                np.sum(self.potential * densities[component])
            )
            rotation -= self.benchmark.rotation * self.weight * float(
                np.real(np.vdot(state, angular))
            )
            effective = sum(
                beta[component, other] * densities[other]
                + dipolar[component, other]
                * dipolar_potentials[other]
                for other in range(2)
            )
            pde_actions.append(
                0.5 * negative_laplacian
                + self.potential * state
                - self.benchmark.rotation * angular
                + effective * state
            )
            for other in range(2):
                interaction += (
                    0.5
                    * self.weight
                    * beta[component, other]
                    * float(
                        np.sum(densities[component] * densities[other])
                    )
                )
                dipolar_energy += (
                    0.5
                    * self.weight
                    * dipolar[component, other]
                    * float(
                        np.sum(
                            densities[component]
                            * dipolar_potentials[other]
                        )
                    )
                )
        chemical_potentials = tuple(
            float(
                self.weight
                * np.real(np.vdot(state, action))
                / mass
            )
            for state, action, mass in zip(
                states, pde_actions, self.benchmark.masses
            )
        )
        energy = (
            kinetic
            + potential_energy
            + rotation
            + interaction
            + dipolar_energy
        )
        virial = (
            2.0 * kinetic
            + 2.0 * interaction
            - 2.0 * potential_energy
            + 3.0 * dipolar_energy
        )
        return (
            float(energy),
            chemical_potentials,
            float(virial),
            dipolar_potentials,
        )

    def initial_state(
        self, kind: str, component: int
    ) -> ComplexArray:
        gaussian = np.exp(-0.5 * (self.x * self.x + self.y * self.y))
        vortex = (self.x + 1j * self.y) * gaussian
        rotation = self.benchmark.rotation
        if kind == "gaussian":
            state = gaussian.astype(complex)
        elif kind == "vortex":
            state = vortex
        elif kind == "conjugate_vortex":
            state = np.conjugate(vortex)
        elif kind == "mixed":
            state = gaussian + vortex
        elif kind == "conjugate_mixed":
            state = gaussian + np.conjugate(vortex)
        elif kind == "rotation_mixed":
            state = (1.0 - rotation) * gaussian + rotation * vortex
        elif kind == "conjugate_rotation_mixed":
            state = np.conjugate(
                (1.0 - rotation) * gaussian + rotation * vortex
            )
        elif kind == "counter_rotation_mixed":
            state = rotation * gaussian + (1.0 - rotation) * vortex
        elif kind == "conjugate_counter_rotation_mixed":
            state = np.conjugate(
                rotation * gaussian + (1.0 - rotation) * vortex
            )
        elif kind == "thomas_fermi":
            beta = np.asarray(self.benchmark.beta)
            effective_beta = float(
                max(beta[component, component], beta[0, 1])
            )
            chemical_potential = 0.5 * np.sqrt(
                4.0 * effective_beta / np.pi
            )
            state = np.sqrt(
                np.maximum(chemical_potential - self.potential, 0.0)
                / effective_beta
            ).astype(complex)
        else:
            raise ValueError(kind)
        return self.normalize(state, self.benchmark.masses[component])


def solve_gflm_ktm(
    benchmark: DipolarBenchmark,
    settings: GFLMKTMSettings,
    initial_kinds: tuple[str, str] = ("gaussian", "gaussian"),
    *,
    callback: HistoryCallback | None = None,
    capture_history: bool = False,
) -> GFLMKTMResult:
    started = perf_counter()
    system = GFLMKTMSpectralSystem(benchmark, settings)
    states = tuple(
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
    emit(
        {
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
        }
    )
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
            stabilization = 0.5 * (
                float(np.max(total_potential))
                + float(np.min(total_potential))
            )
            angular = system.angular_momentum(state)
            right_hand_side = (
                state / settings.time_step
                + (
                    stabilization
                    - total_potential
                    + chemical_potentials[component]
                )
                * state
                + benchmark.rotation * angular
            )
            transformed = np.fft.fft2(right_hand_side)
            candidate = np.fft.ifft2(
                transformed
                / (
                    1.0 / settings.time_step
                    + stabilization
                    + 0.5 * system.wave_number_squared
                )
            )
            candidate = system.normalize(
                candidate, benchmark.masses[component]
            )
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
        emit(
            {
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
                    "converged"
                    if fixed_point_residual <= settings.tolerance
                    else "iterating"
                ),
            }
        )
        if fixed_point_residual <= settings.tolerance:
            break

    converged = fixed_point_residual <= settings.tolerance
    optimizer_status = "converged" if converged else "maximum_iterations"
    optimizer_message = (
        "fixed-point residual reached tolerance"
        if converged
        else "maximum iteration count reached before residual tolerance"
    )
    emit(
        {
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
        }
    )
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
