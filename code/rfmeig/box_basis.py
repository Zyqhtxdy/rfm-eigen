r"""Gaussian-bump random features on a box, for the nonlinear experiments.

Experiments 5 and 6 are posed on a box with a decaying solution rather than on a
bounded domain with a boundary condition, so the boundary factor of the linear
experiments has nothing to attach to.  The features here are instead localized
bumps: a centre drawn uniformly from a box a little smaller than the
computational one, and a width drawn from a small fixed set.  A localized feature
is what a condensate needs -- its density is concentrated where the potential is
low, and a global trigonometric feature would spend most of its mass where the
solution is essentially zero.

The widths are a set rather than a single value because both experiments have
structure on more than one scale: the envelope of the condensate and the
oscillation inside it.  A mixture of widths covers both without tuning either.

The basis reports its values and gradients on request, and everything else in the
nonlinear path -- the energy, the constraint, the Riemannian step -- is written
against that interface.  That is why the same basis serves both a real scalar
state and a two-component complex one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class BoxFeatureBasis:
    bounds: tuple[tuple[float, float], ...]
    kind: str
    parameters: Array
    phases: Array | None = None
    width: float | Array | None = None
    include_constant: bool = True
    hard_boundary: bool = True

    @property
    def dimension(self) -> int:
        return len(self.bounds)

    @property
    def n_features(self) -> int:
        return self.parameters.shape[0] + int(self.include_constant)

    @classmethod
    def sample(
        cls,
        kind: str,
        n_features: int,
        scale: float,
        seed: int,
        bounds: tuple[tuple[float, float], ...],
        *,
        hard_boundary: bool = True,
        center_spread: float | None = None,
    ) -> BoxFeatureBasis:
        if n_features < 2:
            raise ValueError("At least two features are required")
        rng = np.random.default_rng(seed)
        count = n_features - 1
        dimension = len(bounds)
        if kind == "rbf":
            if center_spread is None:
                parameters = rng.uniform(
                    -1.0, 1.0, size=(count, dimension)
                )
            else:
                parameters = np.clip(
                    rng.normal(
                        0.0, center_spread, size=(count, dimension)
                    ),
                    -1.0,
                    1.0,
                )
            return cls(
                bounds=bounds,
                kind=kind,
                parameters=parameters,
                width=float(scale),
                hard_boundary=hard_boundary,
            )
        if kind == "cosine":
            parameters = rng.normal(0.0, scale, size=(count, dimension))
            phases = rng.uniform(0.0, 2.0 * np.pi, size=count)
            return cls(
                bounds=bounds,
                kind=kind,
                parameters=parameters,
                phases=phases,
                hard_boundary=hard_boundary,
            )
        raise ValueError(f"Unknown feature kind: {kind}")

    @classmethod
    def sample_anchored_rbf(
        cls,
        n_features: int,
        width: float,
        seed: int,
        bounds: tuple[tuple[float, float], ...],
        anchors_1d: tuple[float, ...],
        jitter: float,
    ) -> BoxFeatureBasis:
        rng = np.random.default_rng(seed)
        anchor_mesh = np.meshgrid(
            *([np.asarray(anchors_1d)] * len(bounds)), indexing="ij"
        )
        anchors = np.column_stack(
            [coordinates.reshape(-1) for coordinates in anchor_mesh]
        )
        indices = rng.integers(0, anchors.shape[0], size=n_features - 1)
        centers = np.clip(
            anchors[indices]
            + rng.normal(0.0, jitter, size=(n_features - 1, len(bounds))),
            -1.0,
            1.0,
        )
        return cls(
            bounds=bounds,
            kind="rbf",
            parameters=centers,
            width=float(width),
            hard_boundary=True,
        )

    @classmethod
    def sample_multiscale_rbf(
        cls,
        n_features: int,
        widths: tuple[float, ...],
        seed: int,
        bounds: tuple[tuple[float, float], ...],
        center_spread: float,
    ) -> BoxFeatureBasis:
        if n_features < 2 or not widths:
            raise ValueError("Multiscale RBF sampling requires features and widths")
        rng = np.random.default_rng(seed)
        count = n_features - 1
        centers = np.clip(
            rng.normal(
                0.0, center_spread, size=(count, len(bounds))
            ),
            -1.0,
            1.0,
        )
        sampled_widths = np.resize(np.asarray(widths, dtype=float), count)
        rng.shuffle(sampled_widths)
        return cls(
            bounds=bounds,
            kind="rbf",
            parameters=centers,
            width=sampled_widths,
            hard_boundary=True,
        )

    def normalized_coordinates(self, points: Array) -> tuple[Array, Array]:
        midpoint = np.array([(lower + upper) * 0.5 for lower, upper in self.bounds])
        derivative_scale = np.array(
            [2.0 / (upper - lower) for lower, upper in self.bounds]
        )
        return (points - midpoint) * derivative_scale, derivative_scale

    def evaluate(self, points: Array) -> tuple[Array, tuple[Array, ...], Array]:
        coordinates, derivative_scale = self.normalized_coordinates(points)
        phi, phi_gradients, phi_laplacian = self._raw_features(
            coordinates, derivative_scale
        )
        if not self.hard_boundary:
            return phi, phi_gradients, phi_laplacian

        factors = 1.0 - coordinates * coordinates
        boundary = np.prod(factors, axis=1)
        boundary_gradients = []
        boundary_second = []
        for axis in range(self.dimension):
            other = np.prod(np.delete(factors, axis, axis=1), axis=1)
            boundary_gradients.append(
                -2.0
                * coordinates[:, axis]
                * derivative_scale[axis]
                * other
            )
            boundary_second.append(
                -2.0 * derivative_scale[axis] ** 2 * other
            )

        values = boundary[:, None] * phi
        gradients = tuple(
            boundary_gradients[axis][:, None] * phi
            + boundary[:, None] * phi_gradients[axis]
            for axis in range(self.dimension)
        )
        laplacian = boundary[:, None] * phi_laplacian
        for axis in range(self.dimension):
            laplacian += (
                boundary_second[axis][:, None] * phi
                + 2.0
                * boundary_gradients[axis][:, None]
                * phi_gradients[axis]
            )
        return values, gradients, laplacian

    def _raw_features(
        self, coordinates: Array, derivative_scale: Array
    ) -> tuple[Array, tuple[Array, ...], Array]:
        columns = []
        gradients: list[list[Array]] = [
            [] for _ in range(self.dimension)
        ]
        laplacians = []
        if self.include_constant:
            columns.append(np.ones((coordinates.shape[0], 1)))
            for axis in range(self.dimension):
                gradients[axis].append(np.zeros((coordinates.shape[0], 1)))
            laplacians.append(np.zeros((coordinates.shape[0], 1)))

        if self.kind == "rbf":
            if self.width is None:
                raise ValueError("RBF width is missing")
            delta = coordinates[:, None, :] - self.parameters[None, :, :]
            inverse_variance = 1.0 / np.asarray(self.width) ** 2
            phi = np.exp(
                -0.5 * inverse_variance * np.sum(delta * delta, axis=2)
            )
            raw_laplacian = np.zeros_like(phi)
            for axis in range(self.dimension):
                scaled_delta = delta[:, :, axis] * derivative_scale[axis]
                gradients[axis].append(
                    -phi * scaled_delta * inverse_variance
                )
                raw_laplacian += phi * (
                    scaled_delta * scaled_delta * inverse_variance**2
                    - derivative_scale[axis] ** 2 * inverse_variance
                )
        elif self.kind == "cosine":
            if self.phases is None:
                raise ValueError("Cosine phases are missing")
            argument = coordinates @ self.parameters.T + self.phases
            phi = np.cos(argument)
            sine = np.sin(argument)
            squared_frequency = np.zeros(self.parameters.shape[0])
            for axis in range(self.dimension):
                frequency = self.parameters[:, axis] * derivative_scale[axis]
                gradients[axis].append(-sine * frequency)
                squared_frequency += frequency * frequency
            raw_laplacian = -phi * squared_frequency
        else:
            raise ValueError(self.kind)

        columns.append(phi)
        laplacians.append(raw_laplacian)
        return (
            np.concatenate(columns, axis=1),
            tuple(np.concatenate(items, axis=1) for items in gradients),
            np.concatenate(laplacians, axis=1),
        )
