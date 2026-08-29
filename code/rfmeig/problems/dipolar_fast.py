"""Speed-optimized re-implementation of the Example 6 RFM solver.

This module does NOT modify `rsneig`.  It imports the benchmark, the feature
basis and the dipolar kernel unchanged, and re-implements only the parts of
`two_component_dipolar.DipolarRitzSystem` that dominate the run time.

Optimizations, in order of effect:

1.  Real/complex split.  `self.values` is a real (M x r) matrix and the
    coefficients are complex.  `values @ c` therefore promotes the whole
    187 MB matrix to complex128 on every call.  Applying the real matrix to
    the real and imaginary parts separately is mathematically identical and
    avoids both the temporary and half the arithmetic.
2.  A C-contiguous copy of `values.T`, so the adjoint product is a
    contiguous GEMV instead of a strided one.
3.  `operator @ c` is evaluated once per objective call instead of twice.

Nothing else changes: same features, same quadrature, same operator, same
initial data, same optimizer and tolerances.  The computed energies and
chemical potentials must agree with the original implementation to round-off.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.sparse.linalg

from rfmeig.box_basis import Array, BoxFeatureBasis
from rfmeig.problems.dipolar import (
    DipolarBenchmark,
    DipolarResult,
    DipolarSettings,
    HistoryCallback,
    HistoryRecord,
)
from rfmeig.problems.dipolar_kernels import truncated_dipolar_symbol_2d


def _apply_real(matrix: Array, vector: Array) -> Array:
    """matrix (real) @ vector (complex) without complexifying the matrix."""
    return matrix @ vector.real + 1j * (matrix @ vector.imag)


class FastDipolarRitzSystem:
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
        self.transform = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
        self.values = np.ascontiguousarray(raw_values @ self.transform)
        self.values_t = np.ascontiguousarray(self.values.T)
        self.gradients = tuple(
            np.ascontiguousarray(gradient @ self.transform)
            for gradient in raw_gradients
        )
        self.laplacian = np.ascontiguousarray(raw_laplacian @ self.transform)

        potential = 0.5 * np.sum(self.points * self.points, axis=1)
        self.potential = potential
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
        frequencies = 2.0 * np.pi * np.fft.fftfreq(padded_size, d=self.spacing)
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
        self._padded = np.zeros((padded_size, padded_size))

    @property
    def rank(self) -> int:
        return self.retained_rank

    def convolution(self, density: Array) -> Array:
        size = self.grid_size
        self._padded[...] = 0.0
        self._padded[self.padding_slice] = density.reshape(size, size)
        potential = np.fft.ifft2(
            self.dipolar_symbol * np.fft.fft2(self._padded)
        ).real
        return potential[self.padding_slice].reshape(-1)

    def state(self, unit_coefficients: Array):
        blocks = np.split(unit_coefficients, 2)
        coefficients = [
            np.sqrt(mass) * block
            for mass, block in zip(self.benchmark.masses, blocks)
        ]
        values = [_apply_real(self.values, c) for c in coefficients]
        densities = [np.abs(value) ** 2 for value in values]
        return coefficients, values, densities

    def evaluate(self, unit_coefficients: Array):
        coefficients, values, densities = self.state(unit_coefficients)
        dipolar_potentials = [self.convolution(d) for d in densities]
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
        operator_actions = [self.operator @ c for c in coefficients]
        pde_gradients = [
            operator_actions[component]
            + self.weight
            * _apply_real(self.values_t, effective[component] * values[component])
            for component in range(2)
        ]
        chemical_potentials = tuple(
            float(np.real(np.vdot(coefficients[component], pde_gradients[component])))
            / self.benchmark.masses[component]
            for component in range(2)
        )
        energy = sum(
            float(np.real(np.vdot(c, action)))
            for c, action in zip(coefficients, operator_actions)
        )
        for component in range(2):
            for other in range(2):
                energy += 0.5 * self.weight * float(
                    np.dot(
                        densities[component],
                        beta[component, other] * densities[other]
                        + dipolar[component, other] * dipolar_potentials[other],
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
            gradient = gradient - block * np.real(np.vdot(block, gradient))
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
        self, unit_coefficients: Array, chemical_potentials: tuple[float, float]
    ) -> float:
        coefficients, values, densities = self.state(unit_coefficients)
        dipolar_potentials = [self.convolution(d) for d in densities]
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
                -0.5 * _apply_real(self.laplacian, coefficients[component])
                + self.potential * values[component]
                + 1j
                * self.benchmark.rotation
                * (
                    self.points[:, 0]
                    * _apply_real(self.gradients[1], coefficients[component])
                    - self.points[:, 1]
                    * _apply_real(self.gradients[0], coefficients[component])
                )
                + effective * values[component]
                - chemical_potentials[component] * values[component]
            )
            squared += self.weight * float(np.real(np.vdot(residual, residual)))
        volume = np.prod(
            [upper - lower for lower, upper in self.benchmark.bounds]
        )
        return float(np.sqrt(squared / volume))


def solve_fast(
    benchmark: DipolarBenchmark,
    basis: BoxFeatureBasis,
    settings: DipolarSettings,
    initial_seed: int,
    *,
    callback: HistoryCallback | None = None,
    capture_history: bool = False,
) -> DipolarResult:
    started = perf_counter()
    system = FastDipolarRitzSystem(benchmark, basis, settings)
    _, operator_eigenvectors = scipy.linalg.eigh(
        system.operator, subset_by_index=(0, 0), check_finite=False
    )
    initial = operator_eigenvectors[:, 0]
    rng = np.random.default_rng(initial_seed)
    blocks = []
    for _ in range(2):
        noise = rng.normal(size=system.rank) + 1j * rng.normal(size=system.rank)
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
        *,
        phase: str,
        linear_solver_info: int | str = "",
    ) -> tuple[float, tuple[float, float], float]:
        energy, _, chemical_potentials, weak_residual = system.evaluate(
            coefficients
        )
        blocks = np.split(coefficients, 2)
        emit(
            {
                "event": event,
                "phase": phase,
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
                "linear_solver_info": linear_solver_info,
            }
        )
        return energy, chemical_potentials, weak_residual

    record_state(
        "initial", 0, initial_coefficients, "iterating", 0.0, phase="lbfgs"
    )

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
            phase="lbfgs",
        )
        previous_coefficients = coefficients.copy()

    optimizer_started = perf_counter()
    optimization = scipy.optimize.minimize(
        objective,
        real_initial,
        method="L-BFGS-B",
        jac=True,
        callback=iteration_callback,
        options={
            "maxiter": settings.max_iterations,
            # Match the acceptance norm: SciPy tests a real infinity norm,
            # while ``weak_residual`` is a complex projected L2 norm.
            "gtol": settings.residual_tolerance
            / max(4.0, np.sqrt(2.0 * initial_coefficients.size)),
            "ftol": 0.0,
            "maxcor": 20,
            "maxls": settings.maximum_backtracks,
        },
    )
    optimizer_wall_seconds = perf_counter() - optimizer_started
    size = real_initial.size // 2
    raw = optimization.x[:size] + 1j * optimization.x[size:]
    coefficients = system.normalize(raw)
    energy, gradient, chemical_potentials, weak_residual = system.evaluate(
        coefficients
    )
    record_state(
        "optimizer_exit",
        int(optimization.nit),
        coefficients,
        str(optimization.message),
        0.0,
        phase="lbfgs",
    )

    # A successful L-BFGS function-reduction exit is not a convergence proof.
    # If its projected Euler--Lagrange residual is still too large, solve the
    # reference-free stationarity equation with finite-difference
    # Newton--GMRES corrections on the product of complex unit spheres.  The
    # normal and global-phase null directions are explicitly gauged out.
    refinement_started = perf_counter()
    refinement_iterations = 0
    refinement_status: int | str = "not_required"
    refinement_message = "initial L-BFGS residual met tolerance"

    def pack(vector: Array) -> Array:
        return np.concatenate([vector.real, vector.imag])

    def unpack(vector: Array) -> Array:
        half = vector.size // 2
        return vector[:half] + 1j * vector[half:]

    def tangent_projection(vector: Array, state: Array) -> tuple[Array, Array]:
        tangent = vector.copy()
        gauge = np.zeros_like(vector)
        for component in range(2):
            section = slice(component * system.rank, (component + 1) * system.rank)
            block = state[section]
            normal_coefficient = float(
                np.real(np.vdot(block, tangent[section]))
            )
            phase_direction = 1j * block
            phase_coefficient = float(
                np.real(np.vdot(phase_direction, tangent[section]))
            )
            removed = (
                normal_coefficient * block
                + phase_coefficient * phase_direction
            )
            tangent[section] -= removed
            gauge[section] = removed
        return tangent, gauge

    if (
        weak_residual > settings.residual_tolerance
        and settings.refinement_max_iterations > 0
    ):
        refinement_status = "iterating"
        refinement_message = ""
        for _ in range(settings.refinement_max_iterations):
            if weak_residual <= settings.residual_tolerance:
                break
            current = coefficients.copy()
            current_residual = weak_residual
            projected_gradient, _ = tangent_projection(gradient, current)
            dimension = 2 * current.size

            def jacobian_action(real_direction: Array) -> Array:
                complex_direction = unpack(real_direction)
                tangent, gauge = tangent_projection(complex_direction, current)
                norm = float(np.linalg.norm(tangent))
                if norm == 0.0:
                    return pack(gauge)
                step = 2e-6 / max(1.0, norm)
                plus = system.normalize(current + step * tangent)
                minus = system.normalize(current - step * tangent)
                plus_gradient = system.evaluate(plus)[1]
                minus_gradient = system.evaluate(minus)[1]
                derivative = (plus_gradient - minus_gradient) / (2.0 * step)
                derivative, _ = tangent_projection(derivative, current)
                return pack(derivative + gauge)

            jacobian = scipy.sparse.linalg.LinearOperator(
                (dimension, dimension),
                matvec=jacobian_action,
                dtype=float,
            )
            correction, linear_info = scipy.sparse.linalg.gmres(
                jacobian,
                -pack(projected_gradient),
                rtol=min(
                    1e-3,
                    max(
                        1e-7,
                        0.05 * settings.residual_tolerance / weak_residual,
                    ),
                ),
                atol=0.0,
                restart=40,
                maxiter=max(1, settings.refinement_linear_max_iterations // 40),
            )
            direction, _ = tangent_projection(unpack(correction), current)
            if linear_info < 0 or not np.all(np.isfinite(direction)):
                refinement_status = "linear_solve_failed"
                refinement_message = f"GMRES failed with info={linear_info}"
                break
            accepted = False
            step_length = 1.0
            for _ in range(settings.maximum_backtracks):
                candidate = system.normalize(current + step_length * direction)
                candidate_metrics = system.evaluate(candidate)
                if candidate_metrics[3] < current_residual:
                    accepted = True
                    break
                step_length *= 0.5
            if not accepted:
                refinement_status = "residual_line_search_failed"
                refinement_message = "Newton correction did not reduce residual"
                break
            coefficients = candidate
            energy, gradient, chemical_potentials, weak_residual = candidate_metrics
            refinement_iterations += 1
            record_state(
                "iteration",
                int(optimization.nit) + refinement_iterations,
                coefficients,
                (
                    "converged"
                    if weak_residual <= settings.residual_tolerance
                    else "iterating"
                ),
                float(np.linalg.norm(coefficients - current)),
                phase="residual_refinement",
                linear_solver_info=int(linear_info),
            )
        else:
            refinement_status = "maximum_iterations"
            refinement_message = "residual refinement reached iteration limit"
        if weak_residual <= settings.residual_tolerance:
            refinement_status = "converged"
            refinement_message = "projected residual reached tolerance"
    refinement_wall_seconds = perf_counter() - refinement_started
    converged = weak_residual <= settings.residual_tolerance
    record_state(
        "final",
        int(optimization.nit) + refinement_iterations,
        coefficients,
        "converged" if converged else "mathematical_residual_not_met",
        0.0,
        phase="residual_refinement",
    )
    strong_residual_started = perf_counter()
    strong_residual = system.strong_residual(coefficients, chemical_potentials)
    strong_residual_wall_seconds = perf_counter() - strong_residual_started
    _, _, density_vectors = system.state(coefficients)
    densities = tuple(
        d.reshape(system.grid_size, system.grid_size) for d in density_vectors
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
            abs(v - r) / r
            for v, r in zip(chemical_potentials, benchmark.reference_mu)
        ),
        weak_residual=weak_residual,
        pde_residual_rms=strong_residual,
        iterations=int(optimization.nit) + refinement_iterations,
        converged=converged,
        retained_rank=system.retained_rank,
        wall_seconds=perf_counter() - started,
        densities=densities,
        feature_coefficients=feature_coefficients,
        optimizer_success=bool(optimization.success),
        optimizer_status=int(optimization.status),
        optimizer_message=str(optimization.message),
        optimizer_iterations=int(optimization.nit),
        refinement_iterations=refinement_iterations,
        refinement_success=converged,
        refinement_status=refinement_status,
        refinement_message=refinement_message,
        optimizer_wall_seconds=optimizer_wall_seconds,
        refinement_wall_seconds=refinement_wall_seconds,
        strong_residual_wall_seconds=strong_residual_wall_seconds,
        history=history,
    )
