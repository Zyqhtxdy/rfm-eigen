"""Tensor Gauss assembly on the square through one-dimensional Gram products.

For phi_i = x(1-x)y(1-y) cos(omega_i.x + phase_i), expand the cosine of a
sum into two products. The integrals of those products factor by coordinate.
This evaluates the same composite tensor rule without allocating its full
points-by-features-by-dimension array. It is used for Example 1's refined
assembly and H1 error evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rfmeig.problems.separable import sine_basis, sine_basis_derivative
from rfmeig.quadrature import composite_gauss_unit


@dataclass
class SquareForms:
    energy: np.ndarray
    mass: np.ndarray
    h1: np.ndarray
    cross: np.ndarray
    reference_gram: np.ndarray


class SeparableSquareQuadrature:
    """A fixed rule and fixed product reference fields, shared by all draws."""

    def __init__(self, potential, reference_vectors, pairs, order):
        self.order = int(order)
        self.nodes, self.weights = composite_gauss_unit(potential.breakpoints, order)
        self.potential = potential(self.nodes)
        self.pairs = list(pairs)
        modes = reference_vectors.shape[0]
        count = max(max(pair) for pair in pairs) + 1
        self.values = sine_basis(self.nodes, modes) @ reference_vectors[:, :count]
        self.derivatives = sine_basis_derivative(self.nodes, modes) @ reference_vectors[:, :count]
        gram = self.values.T @ (self.weights[:, None] * self.values)
        derivative_gram = self.derivatives.T @ (self.weights[:, None] * self.derivatives)
        self.reference_gram = np.array([
            [gram[a, c] * gram[b, d] + derivative_gram[a, c] * gram[b, d]
             + gram[a, c] * derivative_gram[b, d] for c, d in self.pairs]
            for a, b in self.pairs
        ])

    def assemble(self, omega: np.ndarray, phase: np.ndarray) -> SquareForms:
        x, w = self.nodes, self.weights
        boundary, derivative = x * (1 - x), 1 - 2 * x
        values, derivatives = [], []
        for dim in (0, 1):
            angle = x[:, None] * omega[None, :, dim]
            if dim == 0:
                angle += phase[None, :]
            cosine, sine = np.cos(angle), np.sin(angle)
            values.append([boundary[:, None] * cosine, boundary[:, None] * sine])
            derivatives.append([
                derivative[:, None] * cosine - boundary[:, None] * sine * omega[None, :, dim],
                derivative[:, None] * sine + boundary[:, None] * cosine * omega[None, :, dim],
            ])
        n = len(phase)
        mass, energy, h1 = (np.zeros((n, n)) for _ in range(3))
        for a in (0, 1):
            for b in (0, 1):
                sign = (-1) ** (a + b)
                gx = values[0][a].T @ (w[:, None] * values[0][b])
                gy = values[1][a].T @ (w[:, None] * values[1][b])
                dx = derivatives[0][a].T @ (w[:, None] * derivatives[0][b])
                dy = derivatives[1][a].T @ (w[:, None] * derivatives[1][b])
                vx = values[0][a].T @ ((w * self.potential)[:, None] * values[0][b])
                vy = values[1][a].T @ ((w * self.potential)[:, None] * values[1][b])
                mass += sign * gx * gy
                energy += sign * (dx * gy + gx * dy + vx * gy + gx * vy)
                h1 += sign * (dx * gy + gx * dy + gx * gy)
        cross = np.zeros((len(self.pairs), n))
        for s in (0, 1):
            cx = self.values.T @ (w[:, None] * values[0][s])
            cy = self.values.T @ (w[:, None] * values[1][s])
            dx = self.derivatives.T @ (w[:, None] * derivatives[0][s])
            dy = self.derivatives.T @ (w[:, None] * derivatives[1][s])
            for k, (a, b) in enumerate(self.pairs):
                cross[k] += (-1) ** s * (cx[a] * cy[b] + dx[a] * cy[b] + cx[a] * dy[b])
        return SquareForms(
            (energy + energy.T) / 2, (mass + mass.T) / 2, (h1 + h1.T) / 2,
            cross, self.reference_gram,
        )
