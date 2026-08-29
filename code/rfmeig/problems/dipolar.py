r"""Example 6: a rotating two-component dipolar condensate.

The energy has every term the previous experiment's had and three more, and each
of the three changes the character of the problem.  Rotation enters through an
angular momentum term, which is linear in the state but complex.  The two
components interact through a cross-density term, so neither can be solved alone.
And the dipolar interaction is nonlocal: the potential at a point depends on the
density everywhere, through a convolution.

The nonlocal term is what dictates the implementation.  A convolution is a
product in Fourier space, so the density is transformed, multiplied by the
dipolar symbol of :mod:`rfmeig.problems.dipolar_kernels`, and transformed back;
the cost is that of a transform rather than of a double integral over the box.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import TypeAlias

import numpy as np
import scipy.linalg
import scipy.optimize

from rfmeig.box_basis import Array, BoxFeatureBasis
from rfmeig.problems.dipolar_kernels import truncated_dipolar_symbol_2d

HistoryRecord: TypeAlias = dict[str, float | int | bool | str]
HistoryCallback: TypeAlias = Callable[[HistoryRecord], None]


@dataclass(frozen=True)
class DipolarBenchmark:
    name: str = "E5_two_component_dipolar_2d"
    bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (-16.0, 16.0),
        (-16.0, 16.0),
    )
    masses: tuple[float, float] = (0.1, 0.9)
    beta: tuple[tuple[float, float], tuple[float, float]] = (
        (500.0, 470.0),
        (470.0, 485.0),
    )
    dipolar: tuple[tuple[float, float], tuple[float, float]] = (
        (1.0, 2.0),
        (2.0, 4.0),
    )
    reference_energy: float = 8.3727
    reference_mu: tuple[float, float] = (12.2977, 12.4845)
    rotation: float = 0.1
    dipole_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)


@dataclass(frozen=True)
class DipolarSettings:
    grid_size: int = 96
    mass_cutoff: float = 1e-12
    max_iterations: int = 4000
    residual_tolerance: float = 1e-8
    maximum_backtracks: int = 30
    padding_factor: int = 3
    optimizer: str = "lbfgs"
    refinement_max_iterations: int = 8
    refinement_linear_max_iterations: int = 160


@dataclass
class DipolarResult:
    energy: float
    chemical_potentials: tuple[float, float]
    energy_relative_error: float
    mu_relative_errors: tuple[float, float]
    weak_residual: float
    pde_residual_rms: float
    iterations: int
    converged: bool
    retained_rank: int
    wall_seconds: float
    densities: tuple[Array, Array] | None = None
    feature_coefficients: tuple[Array, Array] | None = None
    optimizer_success: bool = False
    optimizer_status: int | str = "not_started"
    optimizer_message: str = ""
    optimizer_iterations: int = 0
    refinement_iterations: int = 0
    refinement_success: bool = False
    refinement_status: int | str = "not_required"
    refinement_message: str = ""
    optimizer_wall_seconds: float = 0.0
    refinement_wall_seconds: float = 0.0
    strong_residual_wall_seconds: float = 0.0
    history: list[HistoryRecord] | None = None


@dataclass(frozen=True)
class DipolarGridEvaluation:
    grid_size: int
    energy: float
    chemical_potentials: tuple[float, float]
    pde_residual_rms: float
    constraint_errors: tuple[float, float]
    densities: tuple[Array, Array]
    virial_residual: float = 0.0
    kinetic_energy: float = 0.0
    potential_energy: float = 0.0
    rotation_energy: float = 0.0
    interaction_energy: float = 0.0
    dipolar_energy: float = 0.0
    states: tuple[Array, Array] | None = None


class DipolarRitzSystem:
    def __init__(
        self,
        benchmark: DipolarBenchmark,
        basis: BoxFeatureBasis,
        settings: DipolarSettings,
    ) -> None:
        self.benchmark = benchmark
        self.basis = basis
        self.grid_size = settings.grid_size
        lower, upper = benchmark.bounds[0]
        self.spacing = (upper - lower) / settings.grid_size
        nodes = lower + (np.arange(settings.grid_size) + 0.5) * self.spacing
        mesh = np.meshgrid(nodes, nodes, indexing="ij")
        self.points = np.column_stack([axis.reshape(-1) for axis in mesh])
        self.weight = self.spacing**2

        raw_values, raw_gradients, raw_laplacian = basis.evaluate(self.points)
        mass = self.weight * (raw_values.T @ raw_values)
        eigenvalues, eigenvectors = scipy.linalg.eigh(
            mass, check_finite=False, driver="evr"
        )
        keep = eigenvalues > settings.mass_cutoff * eigenvalues[-1]
        self.transform = (
            eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
        )
        self.values = raw_values @ self.transform
        self.gradients = tuple(
            gradient @ self.transform for gradient in raw_gradients
        )
        self.laplacian = raw_laplacian @ self.transform
        potential = 0.5 * np.sum(self.points * self.points, axis=1)
        operator = self.weight * (
            self.values.T @ (potential[:, None] * self.values)
        )
        for gradient in self.gradients:
            operator += 0.5 * self.weight * (gradient.T @ gradient)
        angular_derivative = (
            self.points[:, 0, None] * self.gradients[1]
            - self.points[:, 1, None] * self.gradients[0]
        )
        rotation = (
            1j
            * benchmark.rotation
            * self.weight
            * (self.values.T @ angular_derivative)
        )
        self.operator = 0.5 * (
            operator + operator.T + rotation + rotation.conj().T
        )
        self.retained_rank = int(np.count_nonzero(keep))

        padded_size = settings.padding_factor * settings.grid_size
        frequencies = 2.0 * np.pi * np.fft.fftfreq(
            padded_size, d=self.spacing
        )
        kx, ky = np.meshgrid(frequencies, frequencies, indexing="ij")
        half_width = 0.5 * (upper - lower)
        self.dipolar_symbol = truncated_dipolar_symbol_2d(
            kx,
            ky,
            benchmark.dipole_axis,
            2.0 * np.sqrt(2.0) * half_width,
        )
        offset = (padded_size - settings.grid_size) // 2
        self.padding_slice = (
            slice(offset, offset + settings.grid_size),
            slice(offset, offset + settings.grid_size),
        )

    @property
    def rank(self) -> int:
        return self.retained_rank

    def convolution(self, density: Array) -> Array:
        size = self.grid_size
        padded = np.zeros(self.dipolar_symbol.shape)
        padded[self.padding_slice] = density.reshape(size, size)
        potential = np.fft.ifft2(
            self.dipolar_symbol * np.fft.fft2(padded)
        ).real
        return potential[self.padding_slice].reshape(-1)

    def state(
        self, unit_coefficients: Array
    ) -> tuple[list[Array], list[Array], list[Array]]:
        blocks = np.split(unit_coefficients, 2)
        coefficients = [
            np.sqrt(mass) * block
            for mass, block in zip(self.benchmark.masses, blocks)
        ]
        values = [self.values @ coefficient for coefficient in coefficients]
        densities = [np.abs(value) ** 2 for value in values]
        return coefficients, values, densities

    def evaluate(
        self, unit_coefficients: Array
    ) -> tuple[float, Array, tuple[float, float], float]:
        coefficients, values, densities = self.state(unit_coefficients)
        dipolar_potentials = [
            self.convolution(density) for density in densities
        ]
        beta = np.asarray(self.benchmark.beta)
        dipolar = np.asarray(self.benchmark.dipolar)
        effective = [
            sum(
                beta[component, other] * densities[other]
                + dipolar[component, other] * dipolar_potentials[other]
                for other in range(2)
            )
            for component in range(2)
        ]
        pde_gradients = [
            self.operator @ coefficients[component]
            + self.weight
            * (
                self.values.T
                @ (effective[component] * values[component])
            )
            for component in range(2)
        ]
        chemical_potentials = tuple(
            float(
                np.real(
                    np.vdot(
                        coefficients[component],
                        pde_gradients[component],
                    )
                )
                / self.benchmark.masses[component]
            )
            for component in range(2)
        )
        energy = sum(
            float(np.real(np.vdot(coefficient, self.operator @ coefficient)))
            for coefficient in coefficients
        )
        for component in range(2):
            for other in range(2):
                energy += 0.5 * self.weight * float(
                    np.dot(
                        densities[component],
                        beta[component, other] * densities[other]
                        + dipolar[component, other]
                        * dipolar_potentials[other],
                    )
                )
        gradient_blocks = []
        residual_squared = 0.0
        for component in range(2):
            gradient = (
                np.sqrt(self.benchmark.masses[component])
                * pde_gradients[component]
            )
            block = unit_coefficients[
                component * self.rank : (component + 1) * self.rank
            ]
            gradient -= block * np.real(np.vdot(block, gradient))
            gradient_blocks.append(gradient)
            residual = (
                pde_gradients[component]
                - chemical_potentials[component] * coefficients[component]
            )
            residual_squared += float(np.real(np.vdot(residual, residual)))
        return (
            float(energy),
            np.concatenate(gradient_blocks),
            chemical_potentials,
            float(np.sqrt(residual_squared)),
        )

    def normalize(self, coefficients: Array) -> Array:
        normalized = coefficients.copy()
        for component in range(2):
            block = normalized[
                component * self.rank : (component + 1) * self.rank
            ]
            block /= np.linalg.norm(block)
        return normalized

    def strong_residual(
        self,
        unit_coefficients: Array,
        chemical_potentials: tuple[float, float],
    ) -> float:
        coefficients, values, densities = self.state(unit_coefficients)
        dipolar_potentials = [
            self.convolution(density) for density in densities
        ]
        potential = 0.5 * np.sum(self.points * self.points, axis=1)
        beta = np.asarray(self.benchmark.beta)
        dipolar = np.asarray(self.benchmark.dipolar)
        squared = 0.0
        for component in range(2):
            effective = sum(
                beta[component, other] * densities[other]
                + dipolar[component, other] * dipolar_potentials[other]
                for other in range(2)
            )
            residual = (
                -0.5 * (self.laplacian @ coefficients[component])
                + potential * values[component]
                + 1j
                * self.benchmark.rotation
                * (
                    self.points[:, 0]
                    * (self.gradients[1] @ coefficients[component])
                    - self.points[:, 1]
                    * (self.gradients[0] @ coefficients[component])
                )
                + effective * values[component]
                - chemical_potentials[component] * values[component]
            )
            squared += self.weight * float(np.real(np.vdot(residual, residual)))
        volume = np.prod(
            [upper - lower for lower, upper in self.benchmark.bounds]
        )
        return float(np.sqrt(squared / volume))


def solve_two_component_dipolar(
    benchmark: DipolarBenchmark,
    basis: BoxFeatureBasis,
    settings: DipolarSettings,
    initial_seed: int,
    *,
    callback: HistoryCallback | None = None,
    capture_history: bool = False,
) -> DipolarResult:
    if settings.optimizer == "rcg":
        return _solve_two_component_dipolar_rcg(
            benchmark,
            basis,
            settings,
            initial_seed,
            callback=callback,
            capture_history=capture_history,
        )
    if settings.optimizer != "lbfgs":
        raise ValueError(f"Unknown dipolar optimizer: {settings.optimizer}")

    started = perf_counter()
    system = DipolarRitzSystem(benchmark, basis, settings)
    _, operator_eigenvectors = scipy.linalg.eigh(
        system.operator,
        subset_by_index=(0, 0),
        check_finite=False,
    )
    initial = operator_eigenvectors[:, 0]
    rng = np.random.default_rng(initial_seed)
    blocks = []
    for _ in range(2):
        noise = (
            rng.normal(size=system.rank)
            + 1j * rng.normal(size=system.rank)
        )
        block = initial + 0.08 * noise / np.sqrt(system.rank)
        block /= np.linalg.norm(block)
        blocks.append(block)
    initial_coefficients = np.concatenate(blocks)
    real_initial = np.concatenate(
        [initial_coefficients.real, initial_coefficients.imag]
    )
    history: list[HistoryRecord] | None = [] if capture_history else None
    history_iteration = 0
    previous_coefficients = initial_coefficients.copy()
    objective_evaluations = 0

    def emit(record: HistoryRecord) -> None:
        snapshot = dict(record)
        if history is not None:
            history.append(snapshot)
        if callback is not None:
            callback(dict(snapshot))

    def normalized_state(real_vector: Array) -> Array:
        size = real_vector.size // 2
        raw = real_vector[:size] + 1j * real_vector[size:]
        return system.normalize(raw)

    def record_state(
        event: str,
        iteration: int,
        coefficients: Array,
        optimizer_status: str,
        accepted_step_norm: float,
    ) -> tuple[float, tuple[float, float], float]:
        energy, _, chemical_potentials, weak_residual = system.evaluate(
            coefficients
        )
        blocks = np.split(coefficients, 2)
        emit(
            {
                "event": event,
                "iteration": iteration,
                "cumulative_seconds": perf_counter() - started,
                "energy": energy,
                "mu1": chemical_potentials[0],
                "mu2": chemical_potentials[1],
                "weak_residual": weak_residual,
                "constraint_error_1": abs(
                    float(np.real(np.vdot(blocks[0], blocks[0]))) - 1.0
                ),
                "constraint_error_2": abs(
                    float(np.real(np.vdot(blocks[1], blocks[1]))) - 1.0
                ),
                "step_length": accepted_step_norm,
                "accepted_step_norm": accepted_step_norm,
                "optimizer_function_evaluations": objective_evaluations,
                "optimizer_status": optimizer_status,
            }
        )
        return energy, chemical_potentials, weak_residual

    record_state("initial", 0, initial_coefficients, "iterating", 0.0)

    def objective(real_vector: Array) -> tuple[float, Array]:
        nonlocal objective_evaluations
        objective_evaluations += 1
        size = real_vector.size // 2
        raw = real_vector[:size] + 1j * real_vector[size:]
        coefficients = system.normalize(raw)
        energy, gradient, _, _ = system.evaluate(coefficients)
        scaled_gradient = gradient.copy()
        for component in range(2):
            section = slice(
                component * system.rank, (component + 1) * system.rank
            )
            scaled_gradient[section] /= np.linalg.norm(raw[section])
        return energy, np.concatenate(
            [2.0 * scaled_gradient.real, 2.0 * scaled_gradient.imag]
        )

    def iteration_callback(real_vector: Array) -> None:
        nonlocal history_iteration, previous_coefficients
        history_iteration += 1
        coefficients = normalized_state(real_vector)
        accepted_step_norm = float(
            np.linalg.norm(coefficients - previous_coefficients)
        )
        record_state(
            "iteration",
            history_iteration,
            coefficients,
            "iterating",
            accepted_step_norm,
        )
        previous_coefficients = coefficients.copy()

    optimization = scipy.optimize.minimize(
        objective,
        real_initial,
        method="L-BFGS-B",
        jac=True,
        callback=iteration_callback,
        options={
            "maxiter": settings.max_iterations,
            # L-BFGS-B stops on the infinity norm of a *real* gradient,
            # whereas acceptance below uses the complex projected L2 norm.
            # This scaling prevents a formally successful optimizer exit with
            # a mathematical residual above the requested tolerance.
            "gtol": settings.residual_tolerance
            / max(4.0, np.sqrt(2.0 * initial_coefficients.size)),
            "ftol": 0.0,
            "maxcor": 20,
            "maxls": settings.maximum_backtracks,
        },
    )
    size = real_initial.size // 2
    raw = optimization.x[:size] + 1j * optimization.x[size:]
    coefficients = system.normalize(raw)
    energy, _, chemical_potentials, weak_residual = system.evaluate(coefficients)
    converged = weak_residual <= settings.residual_tolerance
    record_state(
        "final",
        int(optimization.nit),
        coefficients,
        "converged" if converged else "mathematical_residual_not_met",
        0.0,
    )
    strong_residual = system.strong_residual(
        coefficients, chemical_potentials
    )
    return _build_result(
        benchmark=benchmark,
        system=system,
        coefficients=coefficients,
        energy=energy,
        chemical_potentials=chemical_potentials,
        weak_residual=weak_residual,
        strong_residual=strong_residual,
        iterations=int(optimization.nit),
        converged=converged,
        wall_seconds=perf_counter() - started,
        optimizer_success=bool(optimization.success),
        optimizer_status=int(optimization.status),
        optimizer_message=str(optimization.message),
        history=history,
    )


def _solve_two_component_dipolar_rcg(
    benchmark: DipolarBenchmark,
    basis: BoxFeatureBasis,
    settings: DipolarSettings,
    initial_seed: int,
    *,
    callback: HistoryCallback | None = None,
    capture_history: bool = False,
) -> DipolarResult:
    started = perf_counter()
    system = DipolarRitzSystem(benchmark, basis, settings)
    operator_eigenvalues, operator_eigenvectors = scipy.linalg.eigh(
        system.operator, check_finite=False
    )
    inverse_preconditioner = 1.0 / (
        operator_eigenvalues + 1.0 + max(0.0, -operator_eigenvalues[0])
    )
    initial = operator_eigenvectors[:, 0]
    rng = np.random.default_rng(initial_seed)
    blocks = []
    for component in range(2):
        noise = (
            rng.normal(size=system.rank)
            + 1j * rng.normal(size=system.rank)
        )
        block = initial + 0.08 * noise / np.sqrt(system.rank)
        block /= np.linalg.norm(block)
        blocks.append(block)
    coefficients = np.concatenate(blocks)

    def project(vector: Array, state: Array) -> Array:
        result = vector.copy()
        for component in range(2):
            section = slice(component * system.rank, (component + 1) * system.rank)
            result[section] -= (
                state[section]
                * np.real(np.vdot(state[section], result[section]))
            )
        return result

    def precondition(vector: Array, state: Array) -> Array:
        result = np.empty_like(vector)
        for component in range(2):
            section = slice(component * system.rank, (component + 1) * system.rank)
            result[section] = operator_eigenvectors @ (
                inverse_preconditioner
                * (operator_eigenvectors.conj().T @ vector[section])
            )
        return project(result, state)

    energy, gradient, chemical_potentials, weak_residual = system.evaluate(
        coefficients
    )
    preconditioned = precondition(gradient, coefficients)
    direction = -preconditioned
    step_length = 1.0
    history: list[HistoryRecord] | None = [] if capture_history else None
    accepted_iterations = 0
    optimizer_status = "iterating"
    optimizer_message = ""

    def emit(event: str, accepted_step_norm: float) -> None:
        blocks = np.split(coefficients, 2)
        record: HistoryRecord = {
            "event": event,
            "iteration": accepted_iterations,
            "cumulative_seconds": perf_counter() - started,
            "energy": energy,
            "mu1": chemical_potentials[0],
            "mu2": chemical_potentials[1],
            "weak_residual": weak_residual,
            "constraint_error_1": abs(
                float(np.real(np.vdot(blocks[0], blocks[0]))) - 1.0
            ),
            "constraint_error_2": abs(
                float(np.real(np.vdot(blocks[1], blocks[1]))) - 1.0
            ),
            "step_length": step_length if event == "iteration" else 0.0,
            "accepted_step_norm": accepted_step_norm,
            "optimizer_status": optimizer_status,
        }
        if history is not None:
            history.append(dict(record))
        if callback is not None:
            callback(dict(record))

    emit("initial", 0.0)

    for iteration in range(settings.max_iterations):
        if weak_residual <= settings.residual_tolerance:
            optimizer_status = "converged"
            optimizer_message = "projected residual reached tolerance"
            break
        direction = project(direction, coefficients)
        slope = 2.0 * float(np.real(np.vdot(gradient, direction)))
        if not np.isfinite(slope) or slope >= -1e-14:
            direction = -preconditioned
            slope = -2.0 * float(
                np.real(np.vdot(gradient, preconditioned))
            )
        accepted = False
        trial_step = min(2.0, 1.25 * step_length)
        for _ in range(settings.maximum_backtracks):
            candidate = system.normalize(
                coefficients + trial_step * direction
            )
            candidate_energy = system.evaluate(candidate)[0]
            if candidate_energy <= energy + 1e-4 * trial_step * slope:
                accepted = True
                break
            trial_step *= 0.5
        if not accepted:
            optimizer_status = "line_search_failed"
            optimizer_message = "Armijo line search did not accept a step"
            break

        old_gradient = gradient
        old_preconditioned = preconditioned
        old_direction = direction
        old_coefficients = coefficients
        coefficients = candidate
        accepted_iterations += 1
        step_length = trial_step
        (
            energy,
            gradient,
            chemical_potentials,
            weak_residual,
        ) = system.evaluate(coefficients)
        preconditioned = precondition(gradient, coefficients)
        denominator = float(
            np.real(np.vdot(old_gradient, old_preconditioned))
        )
        beta_pr = 0.0
        if denominator > 1e-30:
            beta_pr = float(
                np.real(
                    np.vdot(
                        gradient,
                        preconditioned
                        - project(old_preconditioned, coefficients),
                    )
                )
                / denominator
            )
        if (iteration + 1) % 40 == 0:
            beta_pr = 0.0
        direction = -preconditioned + max(0.0, beta_pr) * project(
            old_direction, coefficients
        )
        emit(
            "iteration",
            float(np.linalg.norm(coefficients - old_coefficients)),
        )
    else:
        optimizer_status = "maximum_iterations"
        optimizer_message = "maximum iteration count reached"

    converged = weak_residual <= settings.residual_tolerance
    if converged:
        optimizer_status = "converged"
        optimizer_message = "projected residual reached tolerance"
    emit("final", 0.0)

    strong_residual = system.strong_residual(
        coefficients, chemical_potentials
    )
    return _build_result(
        benchmark=benchmark,
        system=system,
        coefficients=coefficients,
        energy=energy,
        chemical_potentials=chemical_potentials,
        weak_residual=weak_residual,
        strong_residual=strong_residual,
        iterations=accepted_iterations,
        converged=converged,
        wall_seconds=perf_counter() - started,
        optimizer_success=optimizer_status == "converged",
        optimizer_status=optimizer_status,
        optimizer_message=optimizer_message,
        history=history,
    )


def _build_result(
    *,
    benchmark: DipolarBenchmark,
    system: DipolarRitzSystem,
    coefficients: Array,
    energy: float,
    chemical_potentials: tuple[float, float],
    weak_residual: float,
    strong_residual: float,
    iterations: int,
    converged: bool,
    wall_seconds: float,
    optimizer_success: bool,
    optimizer_status: int | str,
    optimizer_message: str,
    history: list[HistoryRecord] | None,
) -> DipolarResult:
    _, _, density_vectors = system.state(coefficients)
    densities = tuple(
        density.reshape(system.grid_size, system.grid_size)
        for density in density_vectors
    )
    unit_blocks = np.split(coefficients, 2)
    feature_coefficients = tuple(
        np.sqrt(mass) * (system.transform @ block)
        for mass, block in zip(benchmark.masses, unit_blocks)
    )
    return DipolarResult(
        energy=energy,
        chemical_potentials=chemical_potentials,
        energy_relative_error=abs(energy - benchmark.reference_energy)
        / benchmark.reference_energy,
        mu_relative_errors=tuple(
            abs(value - reference) / reference
            for value, reference in zip(
                chemical_potentials, benchmark.reference_mu
            )
        ),
        weak_residual=weak_residual,
        pde_residual_rms=strong_residual,
        iterations=iterations,
        converged=converged,
        retained_rank=system.retained_rank,
        wall_seconds=wall_seconds,
        densities=densities,
        feature_coefficients=feature_coefficients,
        optimizer_success=optimizer_success,
        optimizer_status=optimizer_status,
        optimizer_message=optimizer_message,
        history=history,
    )


def evaluate_dipolar_feature_state(
    benchmark: DipolarBenchmark,
    basis: BoxFeatureBasis,
    feature_coefficients: tuple[Array, Array],
    *,
    grid_size: int,
    padding_factor: int = 3,
    chunk_size: int = 4096,
) -> DipolarGridEvaluation:
    """Evaluate a saved RFM state on an independent midpoint grid.

    Basis matrices are formed in point chunks, so increasing ``grid_size``
    does not require storing a grid-by-feature matrix.  The returned residual
    is a strong PDE residual and is intentionally independent of the assembly
    grid and mass orthogonalization used by the optimizer.
    """
    if grid_size < 4:
        raise ValueError("grid_size must be at least four")
    if padding_factor < 2:
        raise ValueError("padding_factor must be at least two")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    lengths = tuple(upper - lower for lower, upper in benchmark.bounds)
    spacings = tuple(length / grid_size for length in lengths)
    if not np.isclose(spacings[0], spacings[1]):
        raise ValueError("dipolar evaluation currently requires a square mesh")
    nodes = tuple(
        lower + (np.arange(grid_size) + 0.5) * spacing
        for (lower, _), spacing in zip(benchmark.bounds, spacings)
    )
    mesh = np.meshgrid(*nodes, indexing="ij")
    points = np.column_stack([axis.reshape(-1) for axis in mesh])
    count = points.shape[0]
    states = [np.empty(count, dtype=complex) for _ in range(2)]
    gradients = [
        [np.empty(count, dtype=complex) for _ in range(2)]
        for _ in range(2)
    ]
    laplacians = [np.empty(count, dtype=complex) for _ in range(2)]
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        values, derivatives, laplacian = basis.evaluate(points[start:stop])
        for component, coefficients in enumerate(feature_coefficients):
            states[component][start:stop] = values @ coefficients
            for axis in range(2):
                gradients[component][axis][start:stop] = (
                    derivatives[axis] @ coefficients
                )
            laplacians[component][start:stop] = laplacian @ coefficients

    spacing = spacings[0]
    weight = spacings[0] * spacings[1]
    padded_size = padding_factor * grid_size
    frequencies = 2.0 * np.pi * np.fft.fftfreq(padded_size, d=spacing)
    kx, ky = np.meshgrid(frequencies, frequencies, indexing="ij")
    half_width = 0.5 * lengths[0]
    symbol = truncated_dipolar_symbol_2d(
        kx,
        ky,
        benchmark.dipole_axis,
        2.0 * np.sqrt(2.0) * half_width,
    )
    offset = (padded_size - grid_size) // 2
    interior = (
        slice(offset, offset + grid_size),
        slice(offset, offset + grid_size),
    )
    densities = [np.abs(state) ** 2 for state in states]
    dipolar_potentials = []
    for density in densities:
        padded = np.zeros((padded_size, padded_size), dtype=float)
        padded[interior] = density.reshape(grid_size, grid_size)
        potential = np.fft.ifft2(symbol * np.fft.fft2(padded)).real
        dipolar_potentials.append(potential[interior].reshape(-1))

    potential = 0.5 * np.sum(points * points, axis=1)
    angular = [
        points[:, 0] * component_gradients[1]
        - points[:, 1] * component_gradients[0]
        for component_gradients in gradients
    ]
    beta = np.asarray(benchmark.beta)
    dipolar = np.asarray(benchmark.dipolar)
    effective = [
        sum(
            beta[component, other] * densities[other]
            + dipolar[component, other] * dipolar_potentials[other]
            for other in range(2)
        )
        for component in range(2)
    ]
    energy = 0.0
    pde_actions = []
    for component in range(2):
        kinetic = 0.5 * sum(
            np.abs(derivative) ** 2
            for derivative in gradients[component]
        )
        rotation = np.real(
            np.conjugate(states[component])
            * (1j * benchmark.rotation * angular[component])
        )
        energy += weight * float(
            np.sum(kinetic + potential * densities[component] + rotation)
        )
        pde_actions.append(
            -0.5 * laplacians[component]
            + potential * states[component]
            + 1j * benchmark.rotation * angular[component]
            + effective[component] * states[component]
        )
    for component in range(2):
        energy += 0.5 * weight * float(
            np.dot(densities[component], effective[component])
        )
    chemical_potentials = tuple(
        weight
        * float(np.real(np.vdot(states[component], pde_actions[component])))
        / benchmark.masses[component]
        for component in range(2)
    )
    residual_squared = 0.0
    for component in range(2):
        residual = (
            pde_actions[component]
            - chemical_potentials[component] * states[component]
        )
        residual_squared += weight * float(np.real(np.vdot(residual, residual)))
    volume = lengths[0] * lengths[1]
    constraint_errors = tuple(
        abs(weight * float(np.sum(density)) - mass)
        for density, mass in zip(densities, benchmark.masses)
    )
    # The five energy pieces are accumulated separately from ``energy`` so
    # that the value reported above is untouched.  They enter only the virial
    # identity, which Li et al. report as an independent accuracy check that
    # needs no reference solution.  The decomposition and the combination
    # 2 E_kin + 2 E_int - 2 E_pot + 3 E_dip are the ones used by the
    # GFLM--KTM spectral system, so the two methods report the same quantity.
    kinetic_energy = 0.0
    potential_energy = 0.0
    rotation_energy = 0.0
    for component in range(2):
        kinetic_energy += 0.5 * weight * float(
            np.sum(
                sum(
                    np.abs(derivative) ** 2
                    for derivative in gradients[component]
                )
            )
        )
        potential_energy += weight * float(
            np.sum(potential * densities[component])
        )
        rotation_energy += weight * float(
            np.sum(
                np.real(
                    np.conjugate(states[component])
                    * (1j * benchmark.rotation * angular[component])
                )
            )
        )
    interaction_energy = 0.0
    dipolar_energy = 0.0
    for component in range(2):
        for other in range(2):
            interaction_energy += (
                0.5
                * weight
                * float(beta[component, other])
                * float(np.sum(densities[component] * densities[other]))
            )
            dipolar_energy += (
                0.5
                * weight
                * float(dipolar[component, other])
                * float(
                    np.sum(densities[component] * dipolar_potentials[other])
                )
            )
    virial = (
        2.0 * kinetic_energy
        + 2.0 * interaction_energy
        - 2.0 * potential_energy
        + 3.0 * dipolar_energy
    )
    return DipolarGridEvaluation(
        grid_size=grid_size,
        energy=float(energy),
        chemical_potentials=chemical_potentials,
        pde_residual_rms=float(np.sqrt(residual_squared / volume)),
        constraint_errors=constraint_errors,
        densities=tuple(
            density.reshape(grid_size, grid_size) for density in densities
        ),
        virial_residual=abs(float(virial)),
        kinetic_energy=float(kinetic_energy),
        potential_energy=float(potential_energy),
        rotation_energy=float(rotation_energy),
        interaction_energy=float(interaction_energy),
        dipolar_energy=float(dipolar_energy),
        states=tuple(
            state.reshape(grid_size, grid_size) for state in states
        ),
    )


def evaluate_feature_state_fields(
    basis: BoxFeatureBasis,
    feature_coefficients: tuple[Array, Array],
    points: Array,
    *,
    chunk_size: int = 4096,
) -> tuple[Array, Array]:
    """Evaluate the two component fields of an RFM state at given points.

    A random feature state is defined everywhere, so a comparison with a grid
    benchmark can be made on the benchmark's own nodes.  No interpolation
    therefore enters the wave function errors of (4.2) in Li et al.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    nodes = np.asarray(points, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("points must have shape (count, 2)")
    count = nodes.shape[0]
    fields = [np.empty(count, dtype=complex) for _ in range(2)]
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        values, _, _ = basis.evaluate(nodes[start:stop])
        for component, coefficients in enumerate(feature_coefficients):
            fields[component][start:stop] = values @ coefficients
    return (fields[0], fields[1])
