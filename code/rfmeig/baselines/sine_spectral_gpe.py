"""The independent spectral reference for the scalar Gross-Pitaevskii example.

Example 5 measures both the random feature method and the weak Galerkin
baseline against a solution neither of them produces.  That reference was a
pair of constants; this is the solver behind them, so that its own
discretisation error can be read off by refining it.

The ground state of

    -\\Delta u + V u + \\beta |u|^2 u = \\lambda u,   u|_{\\partial\\Omega} = 0,
    \\|u\\|_{L^2(\\Omega)} = 1

on a box is computed in the sine basis, which is the exact eigenbasis of the
Dirichlet Laplacian there, so the only error left is the truncation of the
series -- which is what refining ``modes`` exposes.

The iteration is normalised gradient flow taken implicitly in the whole
operator, the density frozen at the current state:

    (I + h H(u_n)) u* = u_n,   H(u) = -\\Delta + V + \\beta u^2,
    u_{n+1} = u* / \\|u*\\|.

Its fixed point is exact for every step size.  Writing u* = c u at a fixed
point and cancelling gives -\\Delta u + V u + \\beta u^3 = ((1-c)/(ch)) u, so
the step size sets only how fast the iteration arrives, not where.  Taking the
potential or the cubic term explicitly instead leaves the normalisation factor
attached to the Laplacian, which solves a different equation and stalls with a
residual near 1e-2; freezing the density and solving the eigenproblem to
convergence -- a self-consistent field -- diverges at this coupling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dstn, idstn
from scipy.sparse.linalg import LinearOperator, cg

Array = np.ndarray


@dataclass(frozen=True)
class SpectralGroundState:
    """What one solve returns, in the conventions Section 4.2 uses."""

    modes: int
    eigenvalue: float
    energy: float
    kinetic_and_potential: float
    quartic: float
    iterations: int
    stationarity_residual: float

    def as_row(self) -> dict[str, float | int]:
        return {
            "modes": self.modes,
            "lambda": self.eigenvalue,
            "energy": self.energy,
            "iterations": self.iterations,
            "stationarity_residual": self.stationarity_residual,
        }


def _grid(modes: int, half_width: float) -> tuple[Array, Array, float]:
    """The interior collocation points of the sine basis, and their spacing."""
    spacing = 2.0 * half_width / (modes + 1)
    axis = -half_width + spacing * np.arange(1, modes + 1)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    return x, y, spacing


def _laplacian_symbol(modes: int, half_width: float) -> Array:
    """Eigenvalues of the Dirichlet Laplacian on the sine basis."""
    wavenumbers = (np.pi / (2.0 * half_width)) * np.arange(1, modes + 1)
    return wavenumbers[:, None] ** 2 + wavenumbers[None, :] ** 2


def _laplacian(field: Array, symbol: Array) -> Array:
    return idstn(symbol * dstn(field, type=1, norm="ortho"), type=1, norm="ortho")


def solve(
    modes: int,
    *,
    half_width: float = 8.0,
    beta: float = 400.0,
    potential=None,
    step: float = 0.1,
    tolerance: float = 1.0e-13,
    max_iterations: int = 4000,
    linear_tolerance: float = 1.0e-14,
) -> SpectralGroundState:
    """Compute the ground state with ``modes`` sine functions per axis."""
    x, y, spacing = _grid(modes, half_width)
    cell = spacing * spacing
    symbol = _laplacian_symbol(modes, half_width)
    shape = symbol.shape
    size = symbol.size

    if potential is None:
        values = x * x + y * y + 8.0 * np.exp(-((x - 1.0) ** 2) - y * y)
    else:
        values = potential(x, y)

    def normalise(field: Array) -> Array:
        return field / np.sqrt(cell * np.sum(field * field))

    state = normalise(np.exp(-0.25 * (x * x + y * y)))

    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        effective = values + beta * state * state
        # a scalar stand-in for the potential, so the preconditioner stays
        # diagonal in the sine basis
        level = float(np.sum(effective * state * state) * cell)
        inverse = 1.0 / (1.0 + step * (symbol + level))

        def apply(vector: Array, potential_now: Array = effective) -> Array:
            field = vector.reshape(shape)
            return (field + step * (_laplacian(field, symbol)
                                    + potential_now * field)).ravel()

        def precondition(vector: Array, damping: Array = inverse) -> Array:
            return idstn(damping * dstn(vector.reshape(shape), type=1, norm="ortho"),
                         type=1, norm="ortho").ravel()

        operator = LinearOperator((size, size), matvec=apply, dtype=float)
        smoother = LinearOperator((size, size), matvec=precondition, dtype=float)
        solution, _ = cg(operator, state.ravel(), M=smoother,
                         rtol=linear_tolerance, atol=0.0, maxiter=500,
                         x0=state.ravel())

        updated = normalise(solution.reshape(shape))
        change = float(np.max(np.abs(updated - state)))
        state = updated
        if change < tolerance:
            break

    laplacian = _laplacian(state, symbol)
    operator_form = cell * float(np.sum(state * laplacian + values * state * state))
    quartic = cell * float(np.sum(state**4))
    eigenvalue = operator_form + beta * quartic
    residual = laplacian + values * state + beta * state**3 - eigenvalue * state

    return SpectralGroundState(
        modes=modes,
        eigenvalue=eigenvalue,
        energy=0.5 * operator_form + 0.25 * beta * quartic,
        kinetic_and_potential=operator_form,
        quartic=quartic,
        iterations=iterations,
        stationarity_residual=float(np.sqrt(cell * np.sum(residual * residual))),
    )


def refinement_table(mode_counts, **kwargs) -> list[dict[str, float | int]]:
    """Solve at each resolution and report how much the answer still moves."""
    rows = [solve(count, **kwargs).as_row() for count in mode_counts]
    for previous, current in zip(rows, rows[1:], strict=False):
        current["lambda_change"] = abs(current["lambda"] - previous["lambda"])
        current["energy_change"] = abs(current["energy"] - previous["energy"])
    return rows
