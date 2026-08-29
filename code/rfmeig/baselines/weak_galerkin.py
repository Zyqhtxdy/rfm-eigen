r"""The weak Galerkin baseline of Example 5.

An independent reimplementation of the method of Lu and Zhai, not the authors'
program.  The trial space carries a value in each element and a value on each
edge, the gradient is defined weakly against both, and a stabilization term ties
the two together at a scale of :math:`\mathrm{diam}^{-1+\epsilon}`.

That exponent needs a word, because it is the one thing the source does not
disclose.  Lu and Zhai require :math:`\epsilon\in(0,1)` and print no numerical
value.  Taking :math:`\epsilon=0` gives the classical :math:`h^{-1}`
stabilization of weak Galerkin, which their condition weakens rather than
replaces, and that is what these runs use; it reproduces their printed table.
What is claimed is agreement with that table, not knowledge of their exponent.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import TypeAlias

import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray

from rfmeig.problems.condensate import PaperBenchmark

Array = NDArray[np.float64]
HistoryRecord: TypeAlias = dict[str, float | int | bool | str]
HistoryCallback: TypeAlias = Callable[[HistoryRecord], None]


@dataclass(frozen=True)
class WeakGalerkinSettings:
    subdivisions: int
    # See the module docstring: epsilon = 0 is the classical h^{-1}
    # stabilization, and is what reproduces the source's printed table.
    epsilon: float = 0.0
    max_iterations: int = 400
    gradient_tolerance: float = 1e-9
    refinement_max_iterations: int = 20
    refinement_linear_max_iterations: int = 500


@dataclass
class WeakGalerkinResult:
    eigenvalue: float
    energy: float
    residual_norm: float
    iterations: int
    converged: bool
    degrees_of_freedom: int
    wall_seconds: float
    mass_coordinates: Array | None = None
    interior_coefficients: Array | None = None
    edge_coefficients: Array | None = None
    optimizer_success: bool = False
    optimizer_status: int = -1
    optimizer_message: str = ""
    optimizer_iterations: int = 0
    refinement_iterations: int = 0
    refinement_success: bool = False
    refinement_status: int | str = "not_required"
    refinement_message: str = ""
    optimizer_wall_seconds: float = 0.0
    refinement_wall_seconds: float = 0.0
    history: list[HistoryRecord] | None = None


class WeakGalerkinP1System:
    """Lowest-order weak Galerkin discretization used by Lu and Zhai."""

    _quad_phi = np.asarray(
        [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.059715871789770, 0.470142064105115, 0.470142064105115],
            [0.470142064105115, 0.059715871789770, 0.470142064105115],
            [0.470142064105115, 0.470142064105115, 0.059715871789770],
            [0.797426985353087, 0.101286507323456, 0.101286507323456],
            [0.101286507323456, 0.797426985353087, 0.101286507323456],
            [0.101286507323456, 0.101286507323456, 0.797426985353087],
        ]
    )
    _quad_weights = np.asarray(
        [
            0.225000000000000,
            0.132394152788506,
            0.132394152788506,
            0.132394152788506,
            0.125939180544827,
            0.125939180544827,
            0.125939180544827,
        ]
    )
    _local_edges = ((0, 1), (1, 2), (2, 0))

    def __init__(
        self,
        benchmark: PaperBenchmark,
        settings: WeakGalerkinSettings,
    ) -> None:
        if benchmark.dimension != 2:
            raise ValueError("The replicated WG baseline is two-dimensional")
        # diameter**(-1 + epsilon): epsilon=0 is the classical h^{-1}
        # stabilization, and Lu--Zhai weaken it by taking epsilon in (0,1).
        if not 0.0 <= settings.epsilon < 1.0:
            raise ValueError("weak Galerkin epsilon must lie in [0, 1)")
        if settings.refinement_max_iterations < 0:
            raise ValueError("refinement_max_iterations must be nonnegative")
        self.benchmark = benchmark
        self.settings = settings
        (
            self.vertices,
            self.triangles,
            triangle_edges,
            interior_edge_count,
        ) = _uniform_triangular_mesh(
            benchmark.bounds, settings.subdivisions
        )
        self.triangle_edges = triangle_edges
        self.triangle_count = self.triangles.shape[0]
        self.interior_count = 3 * self.triangle_count
        self.edge_count = interior_edge_count

        lower, upper = benchmark.bounds[0]
        mesh_width = (upper - lower) / settings.subdivisions
        self.area = 0.5 * mesh_width * mesh_width
        diameter = np.sqrt(2.0) * mesh_width
        self._assemble(diameter)

        local_mass = (self.area / 12.0) * np.asarray(
            [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]
        )
        self.mass_cholesky = scipy.linalg.cholesky(
            local_mass, lower=True, check_finite=False
        )
        self.mass_inverse_cholesky = scipy.linalg.solve_triangular(
            self.mass_cholesky,
            np.eye(3),
            lower=True,
            check_finite=False,
        )

    @property
    def degrees_of_freedom(self) -> int:
        return self.interior_count + self.edge_count

    def _assemble(self, diameter: float) -> None:
        max_entries = 9 * self.triangle_count
        row00 = np.empty(max_entries, dtype=np.int64)
        col00 = np.empty(max_entries, dtype=np.int64)
        data00 = np.empty(max_entries)
        row0b = np.empty(max_entries, dtype=np.int64)
        col0b = np.empty(max_entries, dtype=np.int64)
        data0b = np.empty(max_entries)
        rowbb = np.empty(max_entries, dtype=np.int64)
        colbb = np.empty(max_entries, dtype=np.int64)
        databb = np.empty(max_entries)
        pointer0b = 0
        pointerbb = 0

        stabilization_scale = diameter ** (-1.0 + self.settings.epsilon)
        for triangle_index, vertex_ids in enumerate(self.triangles):
            points = self.vertices[vertex_ids]
            quadrature_points = self._quad_phi @ points
            potential = self.benchmark.potential(quadrature_points)
            local00 = self.area * (
                self._quad_phi.T
                @ (
                    (self._quad_weights * potential)[:, None]
                    * self._quad_phi
                )
            )
            local0b = np.zeros((3, 3))
            localbb = np.zeros((3, 3))
            normal_lengths = []

            for local_edge, (left, right) in enumerate(self._local_edges):
                tangent = points[right] - points[left]
                edge_length = float(np.linalg.norm(tangent))
                normal_lengths.append(
                    np.asarray([tangent[1], -tangent[0]])
                )
                coefficient = stabilization_scale * edge_length
                edge_average = np.zeros(3)
                edge_average[left] = 0.5
                edge_average[right] = 0.5
                local00 += coefficient * np.outer(
                    edge_average, edge_average
                )
                if self.triangle_edges[triangle_index, local_edge] >= 0:
                    local0b[:, local_edge] -= coefficient * edge_average
                    localbb[local_edge, local_edge] += coefficient

            normal_lengths_array = np.asarray(normal_lengths)
            localbb += (
                normal_lengths_array @ normal_lengths_array.T / self.area
            )

            interior_ids = np.arange(
                3 * triangle_index, 3 * triangle_index + 3
            )
            block = slice(9 * triangle_index, 9 * triangle_index + 9)
            row00[block] = np.repeat(interior_ids, 3)
            col00[block] = np.tile(interior_ids, 3)
            data00[block] = local00.reshape(-1)

            for local_edge, edge_id in enumerate(
                self.triangle_edges[triangle_index]
            ):
                if edge_id < 0:
                    continue
                values = local0b[:, local_edge]
                block0b = slice(pointer0b, pointer0b + 3)
                row0b[block0b] = interior_ids
                col0b[block0b] = edge_id
                data0b[block0b] = values
                pointer0b += 3

            active = np.flatnonzero(
                self.triangle_edges[triangle_index] >= 0
            )
            for left in active:
                for right in active:
                    rowbb[pointerbb] = self.triangle_edges[
                        triangle_index, left
                    ]
                    colbb[pointerbb] = self.triangle_edges[
                        triangle_index, right
                    ]
                    databb[pointerbb] = localbb[left, right]
                    pointerbb += 1

        self.matrix00 = scipy.sparse.coo_matrix(
            (data00, (row00, col00)),
            shape=(self.interior_count, self.interior_count),
        ).tocsr()
        self.matrix0b = scipy.sparse.coo_matrix(
            (
                data0b[:pointer0b],
                (row0b[:pointer0b], col0b[:pointer0b]),
            ),
            shape=(self.interior_count, self.edge_count),
        ).tocsr()
        matrixbb = scipy.sparse.coo_matrix(
            (
                databb[:pointerbb],
                (rowbb[:pointerbb], colbb[:pointerbb]),
            ),
            shape=(self.edge_count, self.edge_count),
        ).tocsc()
        self.edge_factor = scipy.sparse.linalg.splu(matrixbb)

    def _mass_to_interior(self, mass_coordinates: Array) -> Array:
        blocks = mass_coordinates.reshape(-1, 3)
        return (blocks @ self.mass_inverse_cholesky).reshape(-1)

    def _interior_to_mass_gradient(self, gradient: Array) -> Array:
        blocks = gradient.reshape(-1, 3)
        return (
            blocks @ self.mass_inverse_cholesky.T
        ).reshape(-1)

    def _schur_action(self, interior: Array) -> Array:
        edge = -self.edge_factor.solve(self.matrix0b.T @ interior)
        return self.matrix00 @ interior + self.matrix0b @ edge

    def evaluate(self, mass_coordinates: Array) -> tuple[float, Array, float]:
        interior = self._mass_to_interior(mass_coordinates)
        linear_action = self._schur_action(interior)
        local_values = interior.reshape(-1, 3) @ self._quad_phi.T
        weighted_cube = (
            self.area
            * self._quad_weights[None, :]
            * local_values**3
        )
        nonlinear_gradient = self.benchmark.beta * (
            weighted_cube @ self._quad_phi
        ).reshape(-1)
        quartic = self.area * float(
            np.sum(
                self._quad_weights[None, :] * local_values**4
            )
        )
        energy = (
            0.5 * float(interior @ linear_action)
            + 0.25 * self.benchmark.beta * quartic
        )
        gradient = self._interior_to_mass_gradient(
            linear_action + nonlinear_gradient
        )
        eigenvalue = float(mass_coordinates @ gradient)
        return energy, gradient, eigenvalue

    def hessian_action(
        self, mass_coordinates: Array, direction: Array
    ) -> Array:
        """Apply the exact Euclidean Hessian in mass coordinates."""
        interior = self._mass_to_interior(mass_coordinates)
        interior_direction = self._mass_to_interior(direction)
        local_values = interior.reshape(-1, 3) @ self._quad_phi.T
        local_direction = (
            interior_direction.reshape(-1, 3) @ self._quad_phi.T
        )
        weighted_variation = (
            self.area
            * self._quad_weights[None, :]
            * 3.0
            * local_values**2
            * local_direction
        )
        nonlinear_action = self.benchmark.beta * (
            weighted_variation @ self._quad_phi
        ).reshape(-1)
        return self._interior_to_mass_gradient(
            self._schur_action(interior_direction) + nonlinear_action
        )

    def initial_state(self) -> Array:
        points = self.vertices[self.triangles].reshape(-1, 2)
        shifted = points - np.asarray([0.25, 0.0])
        interior = np.exp(-0.5 * np.sum(shifted * shifted, axis=1))
        mass_coordinates = (
            interior.reshape(-1, 3) @ self.mass_cholesky
        ).reshape(-1)
        return mass_coordinates / np.linalg.norm(mass_coordinates)


def solve_weak_galerkin(
    benchmark: PaperBenchmark,
    settings: WeakGalerkinSettings,
    *,
    callback: HistoryCallback | None = None,
    capture_history: bool = False,
) -> WeakGalerkinResult:
    started = perf_counter()
    system = WeakGalerkinP1System(benchmark, settings)
    initial = system.initial_state()
    history: list[HistoryRecord] | None = [] if capture_history else None
    history_iteration = 0
    previous_state = initial.copy()

    def emit(record: HistoryRecord) -> None:
        snapshot = dict(record)
        if history is not None:
            history.append(snapshot)
        if callback is not None:
            callback(dict(snapshot))

    def state_metrics(
        variable: Array,
    ) -> tuple[Array, float, Array, float, float, float]:
        norm = np.linalg.norm(variable)
        state = variable / norm
        energy, gradient, eigenvalue = system.evaluate(state)
        residual = float(np.linalg.norm(gradient - eigenvalue * state))
        constraint_error = abs(float(state @ state) - 1.0)
        return state, energy, gradient, eigenvalue, residual, constraint_error

    (
        _,
        initial_energy,
        _,
        initial_eigenvalue,
        initial_residual,
        initial_constraint_error,
    ) = state_metrics(initial)
    emit(
        {
            "event": "initial",
            "phase": "lbfgs",
            "iteration": 0,
            "cumulative_seconds": perf_counter() - started,
            "energy": initial_energy,
            "eigenvalue": initial_eigenvalue,
            "weak_residual": initial_residual,
            "constraint_error": initial_constraint_error,
            "step_length": 0.0,
            "accepted_step_norm": 0.0,
            "optimizer_status": "iterating",
        }
    )

    def objective(variable: Array) -> tuple[float, Array]:
        norm = np.linalg.norm(variable)
        state = variable / norm
        energy, gradient, eigenvalue = system.evaluate(state)
        tangent = gradient - eigenvalue * state
        return energy, tangent / norm

    def iteration_callback(variable: Array) -> None:
        nonlocal history_iteration, previous_state
        history_iteration += 1
        state, energy, _, eigenvalue, residual, constraint_error = state_metrics(
            variable
        )
        accepted_step_norm = float(np.linalg.norm(state - previous_state))
        previous_state = state.copy()
        emit(
            {
                "event": "iteration",
                "phase": "lbfgs",
                "iteration": history_iteration,
                "cumulative_seconds": perf_counter() - started,
                "energy": energy,
                "eigenvalue": eigenvalue,
                "weak_residual": residual,
                "constraint_error": constraint_error,
                "step_length": accepted_step_norm,
                "accepted_step_norm": accepted_step_norm,
                "optimizer_status": "iterating",
            }
        )

    optimizer_started = perf_counter()
    optimization = scipy.optimize.minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        callback=iteration_callback,
        options={
            "maxiter": settings.max_iterations,
            "gtol": settings.gradient_tolerance,
            "ftol": 1e-15,
            "maxcor": 12,
            "maxls": 30,
        },
    )
    optimizer_wall_seconds = perf_counter() - optimizer_started
    state, energy, gradient, eigenvalue, last_residual, constraint_error = (
        state_metrics(optimization.x)
    )
    emit(
        {
            "event": "optimizer_exit",
            "phase": "lbfgs",
            "iteration": int(optimization.nit),
            "cumulative_seconds": perf_counter() - started,
            "energy": energy,
            "eigenvalue": eigenvalue,
            "weak_residual": last_residual,
            "constraint_error": constraint_error,
            "step_length": 0.0,
            "accepted_step_norm": 0.0,
            "optimizer_status": str(optimization.message),
        }
    )

    # L-BFGS-B can report a successful function-reduction exit while the
    # projected Euler--Lagrange residual is still too large.  Continue with a
    # constrained Newton--MINRES correction.  Both its right-hand side and its
    # line-search merit function are the reference-free Euler--Lagrange
    # residual; no exact or published solution is consulted.
    refinement_iterations = 0
    refinement_wall_seconds = 0.0
    refinement_status: int | str = "not_required"
    refinement_message = "initial L-BFGS residual met tolerance"
    if (
        last_residual > settings.gradient_tolerance
        and settings.refinement_max_iterations > 0
    ):
        refinement_started = perf_counter()
        refinement_status = "iterating"
        refinement_message = ""
        for _ in range(settings.refinement_max_iterations):
            if last_residual <= settings.gradient_tolerance:
                refinement_status = "converged"
                refinement_message = "projected residual reached tolerance"
                break
            rank = state.size

            def saddle_action(vector: Array) -> Array:
                tangent = vector[:-1]
                multiplier = vector[-1]
                return np.append(
                    system.hessian_action(state, tangent)
                    - eigenvalue * tangent
                    - multiplier * state,
                    float(state @ tangent),
                )

            saddle = scipy.sparse.linalg.LinearOperator(
                (rank + 1, rank + 1), matvec=saddle_action, dtype=float
            )
            forcing = min(
                1e-4,
                max(1e-12, 0.05 * settings.gradient_tolerance / last_residual),
            )
            correction, linear_info = scipy.sparse.linalg.minres(
                saddle,
                np.append(-(gradient - eigenvalue * state), 0.0),
                rtol=forcing,
                maxiter=settings.refinement_linear_max_iterations,
                show=False,
                check=False,
            )
            direction = correction[:-1]
            if linear_info < 0 or not np.all(np.isfinite(direction)):
                refinement_status = "linear_solve_failed"
                refinement_message = f"MINRES failed with info={linear_info}"
                break
            direction -= state * float(state @ direction)
            step_length = 1.0
            accepted = False
            previous_state = state.copy()
            previous_residual = last_residual
            for _ in range(30):
                candidate = state + step_length * direction
                candidate /= np.linalg.norm(candidate)
                candidate_metrics = state_metrics(candidate)
                candidate_residual = candidate_metrics[4]
                if candidate_residual < previous_residual:
                    accepted = True
                    break
                step_length *= 0.5
            if not accepted:
                refinement_status = "residual_line_search_failed"
                refinement_message = "Newton correction did not reduce residual"
                break
            (
                state,
                energy,
                gradient,
                eigenvalue,
                last_residual,
                constraint_error,
            ) = candidate_metrics
            refinement_iterations += 1
            emit(
                {
                    "event": "iteration",
                    "phase": "residual_refinement",
                    "iteration": int(optimization.nit) + refinement_iterations,
                    "cumulative_seconds": perf_counter() - started,
                    "energy": energy,
                    "eigenvalue": eigenvalue,
                    "weak_residual": last_residual,
                    "constraint_error": constraint_error,
                    "step_length": step_length,
                    "accepted_step_norm": float(
                        np.linalg.norm(state - previous_state)
                    ),
                    "linear_solver_info": int(linear_info),
                    "optimizer_status": (
                        "converged"
                        if last_residual <= settings.gradient_tolerance
                        else "iterating"
                    ),
                }
            )
        else:
            refinement_status = "maximum_iterations"
            refinement_message = "residual refinement reached iteration limit"
        refinement_wall_seconds = perf_counter() - refinement_started
        if last_residual <= settings.gradient_tolerance:
            refinement_status = "converged"
            refinement_message = "projected residual reached tolerance"

    converged = last_residual <= settings.gradient_tolerance
    interior_coefficients = system._mass_to_interior(state)
    edge_coefficients = -system.edge_factor.solve(
        system.matrix0b.T @ interior_coefficients
    )
    emit(
        {
            "event": "final",
            "phase": "residual_refinement",
            "iteration": int(optimization.nit) + refinement_iterations,
            "cumulative_seconds": perf_counter() - started,
            "energy": energy,
            "eigenvalue": eigenvalue,
            "weak_residual": last_residual,
            "constraint_error": constraint_error,
            "step_length": 0.0,
            "accepted_step_norm": 0.0,
            "optimizer_status": (
                "converged" if converged else "mathematical_residual_not_met"
            ),
        }
    )
    return WeakGalerkinResult(
        eigenvalue=eigenvalue,
        energy=energy,
        residual_norm=last_residual,
        iterations=int(optimization.nit) + refinement_iterations,
        converged=converged,
        degrees_of_freedom=system.degrees_of_freedom,
        wall_seconds=perf_counter() - started,
        mass_coordinates=state,
        interior_coefficients=interior_coefficients,
        edge_coefficients=edge_coefficients,
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
        history=history,
    )


def _uniform_triangular_mesh(
    bounds: tuple[tuple[float, float], ...],
    subdivisions: int,
) -> tuple[Array, NDArray[np.int64], NDArray[np.int64], int]:
    (x_lower, x_upper), (y_lower, y_upper) = bounds
    x = np.linspace(x_lower, x_upper, subdivisions + 1)
    y = np.linspace(y_lower, y_upper, subdivisions + 1)
    mesh_x, mesh_y = np.meshgrid(x, y, indexing="ij")
    vertices = np.column_stack([mesh_x.reshape(-1), mesh_y.reshape(-1)])

    def vertex(i: int, j: int) -> int:
        return i * (subdivisions + 1) + j

    triangles = []
    for i in range(subdivisions):
        for j in range(subdivisions):
            lower_left = vertex(i, j)
            lower_right = vertex(i + 1, j)
            upper_left = vertex(i, j + 1)
            upper_right = vertex(i + 1, j + 1)
            # Lu--Zhai's published mesh figure uses the upper-left to
            # lower-right diagonal in every square.  Both triangles below are
            # counter-clockwise in the (x, y) coordinates used here.
            triangles.append((lower_left, lower_right, upper_left))
            triangles.append((lower_right, upper_right, upper_left))
    triangle_array = np.asarray(triangles, dtype=np.int64)

    edge_occurrences: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for triangle_index, triangle in enumerate(triangle_array):
        for local_edge, (left, right) in enumerate(
            WeakGalerkinP1System._local_edges
        ):
            edge = tuple(sorted((int(triangle[left]), int(triangle[right]))))
            edge_occurrences.setdefault(edge, []).append(
                (triangle_index, local_edge)
            )

    triangle_edges = np.full((triangle_array.shape[0], 3), -1, dtype=np.int64)
    interior_edge_id = 0
    for occurrences in edge_occurrences.values():
        if len(occurrences) != 2:
            continue
        for triangle_index, local_edge in occurrences:
            triangle_edges[triangle_index, local_edge] = interior_edge_id
        interior_edge_id += 1
    return vertices, triangle_array, triangle_edges, interior_edge_id
