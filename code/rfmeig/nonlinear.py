r"""The Riemannian solver for a real nonlinear eigenvalue problem.

Example 5 minimizes the Gross--Pitaevskii energy

.. math::

    E[u]=\int \tfrac12|\nabla u|^2+Vu^2+\tfrac{\beta}{2}u^4

over the unit sphere of :math:`L^2`.  The constraint is what makes this a
Riemannian problem rather than an unconstrained one: the admissible set is a
sphere, the gradient has to be projected onto its tangent space, and the step has
to be retracted back onto it.  Doing that, rather than adding a penalty, keeps
the mass exactly one at every iterate and makes the reported eigenvalue exactly
the Lagrange multiplier of the constraint.

Nothing in Section 3 covers this.  The theory there is for the linear
self-adjoint Dirichlet problem, and the manuscript says so; what carries
over is the trial space of Section 2.3, not the estimates.

The discrete state is a coefficient vector against the feature basis, so the mass
matrix is not the identity and the sphere is an ellipsoid in coefficient
coordinates.  It is made a sphere by whitening that matrix, with the same
truncation of unresolved directions the linear experiments use.

Two residuals are reported, and they measure different things.  The weak residual
is of the discrete equations, and a solver that has converged inside a space too
small to hold the solution can drive it to zero.  The pointwise residual of the
differential equation is evaluated on a finer rule than the one the problem was
assembled on, and is the one that would expose exactly that.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import TypeAlias

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from rfmeig.box_basis import BoxFeatureBasis
from rfmeig.problems.condensate import PaperBenchmark

Array = NDArray[np.float64]
HistoryRecord: TypeAlias = dict[str, float | int | bool | str]
HistoryCallback: TypeAlias = Callable[[HistoryRecord], None]


@dataclass(frozen=True)
class RiemannianSettings:
    quadrature_order: int
    evaluation_order: int
    mass_cutoff: float = 1e-12
    max_iterations: int = 100
    residual_tolerance: float = 1e-10
    maximum_backtracks: int = 30


@dataclass
class RiemannianResult:
    coefficients: Array
    eigenvalue: float
    energy: float
    weak_residual: float
    pde_residual_rms: float
    iterations: int
    converged: bool
    retained_rank: int
    wall_seconds: float
    optimizer_success: bool = False
    optimizer_status: str = "not_started"
    optimizer_message: str = ""
    history: list[HistoryRecord] | None = None


def tensor_gauss_legendre_nd(
    order: int, bounds: tuple[tuple[float, float], ...]
) -> tuple[Array, Array]:
    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(order)
    transformed_nodes = []
    transformed_weights = []
    for lower, upper in bounds:
        transformed_nodes.append(
            0.5 * (upper - lower) * nodes_1d + 0.5 * (upper + lower)
        )
        transformed_weights.append(0.5 * (upper - lower) * weights_1d)
    node_mesh = np.meshgrid(*transformed_nodes, indexing="ij")
    weight_mesh = np.meshgrid(*transformed_weights, indexing="ij")
    points = np.column_stack([mesh.reshape(-1) for mesh in node_mesh])
    weights = np.prod(
        np.stack(weight_mesh, axis=-1), axis=-1
    ).reshape(-1)
    return points, weights


class ScalarRitzSystem:
    def __init__(
        self,
        benchmark: PaperBenchmark,
        basis: BoxFeatureBasis,
        order: int,
        mass_cutoff: float,
    ) -> None:
        self.benchmark = benchmark
        self.basis = basis
        self.points, self.weights = tensor_gauss_legendre_nd(
            order, benchmark.bounds
        )
        raw_values, raw_gradients, _ = basis.evaluate(self.points)
        root_weight = np.sqrt(self.weights)
        weighted_values = root_weight[:, None] * raw_values
        mass = weighted_values.T @ weighted_values
        eigenvalues, eigenvectors = scipy.linalg.eigh(
            mass, check_finite=False, driver="evr"
        )
        keep = eigenvalues > mass_cutoff * eigenvalues[-1]
        if not np.any(keep):
            raise RuntimeError("No numerically independent random feature remains")
        self.transform = (
            eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
        )
        self.values = raw_values @ self.transform
        self.gradients = tuple(
            gradient @ self.transform for gradient in raw_gradients
        )
        weighted_transformed = root_weight[:, None] * self.values
        operator = weighted_transformed.T @ (
            benchmark.potential(self.points)[:, None] * weighted_transformed
        )
        for gradient in self.gradients:
            weighted_gradient = root_weight[:, None] * gradient
            operator += weighted_gradient.T @ weighted_gradient
        self.operator = 0.5 * (operator + operator.T)
        self.retained_rank = int(np.count_nonzero(keep))

    def nonlinear_terms(self, coefficients: Array) -> tuple[Array, Array]:
        values = self.values @ coefficients
        weighted_square = (self.weights * values * values)[:, None]
        residual = self.benchmark.beta * (
            self.values.T @ (self.weights * values**3)
        )
        jacobian = 3.0 * self.benchmark.beta * (
            self.values.T @ (weighted_square * self.values)
        )
        return residual, jacobian

    def energy(self, coefficients: Array) -> float:
        values = self.values @ coefficients
        return float(
            0.5 * coefficients @ self.operator @ coefficients
            + 0.25
            * self.benchmark.beta
            * np.dot(self.weights, values**4)
        )

    def rayleigh(self, coefficients: Array) -> float:
        nonlinear, _ = self.nonlinear_terms(coefficients)
        return float(coefficients @ (self.operator @ coefficients + nonlinear))


def solve_scalar_riemannian(
    benchmark: PaperBenchmark,
    basis: BoxFeatureBasis,
    settings: RiemannianSettings,
    *,
    callback: HistoryCallback | None = None,
    capture_history: bool = False,
) -> RiemannianResult:
    started = perf_counter()
    system = ScalarRitzSystem(
        benchmark, basis, settings.quadrature_order, settings.mass_cutoff
    )
    _, eigenvectors = scipy.linalg.eigh(
        system.operator,
        subset_by_index=(0, 0),
        check_finite=False,
    )
    coefficients = eigenvectors[:, 0]
    history: list[HistoryRecord] | None = [] if capture_history else None

    def emit(record: HistoryRecord) -> None:
        snapshot = dict(record)
        if history is not None:
            history.append(snapshot)
        if callback is not None:
            callback(dict(snapshot))

    def metrics(state: Array) -> tuple[Array, Array, float, Array, float, float]:
        nonlinear, nonlinear_jacobian = system.nonlinear_terms(state)
        gradient = system.operator @ state + nonlinear
        value = float(state @ gradient)
        residual = gradient - value * state
        residual_norm = float(np.linalg.norm(residual))
        energy = system.energy(state)
        return nonlinear_jacobian, gradient, value, residual, residual_norm, energy

    (
        nonlinear_jacobian,
        gradient,
        eigenvalue,
        residual,
        residual_norm,
        energy,
    ) = metrics(coefficients)
    accepted_iterations = 0
    optimizer_status = "iterating"
    optimizer_message = ""
    emit(
        {
            "event": "initial",
            "iteration": 0,
            "cumulative_seconds": perf_counter() - started,
            "energy": energy,
            "eigenvalue": eigenvalue,
            "weak_residual": residual_norm,
            "constraint_error": abs(float(coefficients @ coefficients) - 1.0),
            "step_length": 0.0,
            "accepted_step_norm": 0.0,
            "backtracks": 0,
            "optimizer_status": optimizer_status,
        }
    )

    for _ in range(settings.max_iterations):
        if residual_norm <= settings.residual_tolerance:
            optimizer_status = "converged"
            optimizer_message = "projected residual reached tolerance"
            break

        rank = coefficients.size
        saddle = np.empty((rank + 1, rank + 1))
        saddle[:-1, :-1] = (
            system.operator
            + nonlinear_jacobian
            - eigenvalue * np.eye(rank)
        )
        saddle[:-1, -1] = -coefficients
        saddle[-1, :-1] = coefficients
        saddle[-1, -1] = 0.0
        right_hand_side = np.append(-residual, 0.0)
        try:
            direction = scipy.linalg.solve(
                saddle, right_hand_side, check_finite=False
            )[:-1]
        except scipy.linalg.LinAlgError:
            direction = scipy.linalg.lstsq(
                saddle,
                right_hand_side,
                cond=1e-13,
                lapack_driver="gelsy",
                check_finite=False,
            )[0][:-1]

        slope = float(gradient @ direction)
        if not np.isfinite(slope) or slope >= 0.0:
            direction = -residual
            slope = -(residual_norm**2)

        step_length = 1.0
        accepted = False
        backtracks = 0
        old_coefficients = coefficients
        for backtracks in range(settings.maximum_backtracks):
            candidate = coefficients + step_length * direction
            candidate /= np.linalg.norm(candidate)
            if system.energy(candidate) <= (
                energy + 1e-4 * step_length * slope
            ):
                coefficients = candidate
                accepted = True
                break
            step_length *= 0.5
        if not accepted:
            optimizer_status = "line_search_failed"
            optimizer_message = "Armijo line search did not accept a step"
            break
        coefficients = candidate
        accepted_iterations += 1
        (
            nonlinear_jacobian,
            gradient,
            eigenvalue,
            residual,
            residual_norm,
            energy,
        ) = metrics(coefficients)
        emit(
            {
                "event": "iteration",
                "iteration": accepted_iterations,
                "cumulative_seconds": perf_counter() - started,
                "energy": energy,
                "eigenvalue": eigenvalue,
                "weak_residual": residual_norm,
                "constraint_error": abs(float(coefficients @ coefficients) - 1.0),
                "step_length": step_length,
                "accepted_step_norm": float(
                    np.linalg.norm(coefficients - old_coefficients)
                ),
                "backtracks": backtracks,
                "optimizer_status": (
                    "converged"
                    if residual_norm <= settings.residual_tolerance
                    else "iterating"
                ),
            }
        )
    else:
        optimizer_status = "maximum_iterations"
        optimizer_message = "maximum iteration count reached"

    converged = residual_norm <= settings.residual_tolerance
    if converged:
        optimizer_status = "converged"
        optimizer_message = "projected residual reached tolerance"
    emit(
        {
            "event": "final",
            "iteration": accepted_iterations,
            "cumulative_seconds": perf_counter() - started,
            "energy": energy,
            "eigenvalue": eigenvalue,
            "weak_residual": residual_norm,
            "constraint_error": abs(float(coefficients @ coefficients) - 1.0),
            "step_length": 0.0,
            "accepted_step_norm": 0.0,
            "backtracks": 0,
            "optimizer_status": optimizer_status,
        }
    )

    physical_coefficients = system.transform @ coefficients
    eigenvalue = system.rayleigh(coefficients)
    energy = system.energy(coefficients)
    pde_residual = evaluate_pde_residual(
        benchmark,
        basis,
        physical_coefficients,
        eigenvalue,
        settings.evaluation_order,
    )
    return RiemannianResult(
        coefficients=physical_coefficients,
        eigenvalue=eigenvalue,
        energy=energy,
        weak_residual=residual_norm,
        pde_residual_rms=pde_residual,
        iterations=accepted_iterations,
        converged=converged,
        retained_rank=system.retained_rank,
        wall_seconds=perf_counter() - started,
        optimizer_success=converged,
        optimizer_status=optimizer_status,
        optimizer_message=optimizer_message,
        history=history,
    )


def evaluate_pde_residual(
    benchmark: PaperBenchmark,
    basis: BoxFeatureBasis,
    coefficients: Array,
    eigenvalue: float,
    order: int,
) -> float:
    points, weights = tensor_gauss_legendre_nd(order, benchmark.bounds)
    values, _, laplacian = basis.evaluate(points)
    solution = values @ coefficients
    residual = (
        -(laplacian @ coefficients)
        + benchmark.potential(points) * solution
        + benchmark.beta * solution**3
        - eigenvalue * solution
    )
    volume = float(np.prod([upper - lower for lower, upper in benchmark.bounds]))
    return float(np.sqrt(np.dot(weights, residual * residual) / volume))
