r"""Stored Deep Ritz networks for the ten-dimensional benchmark of Ji et al.

The reported estimator is (domain_energy + eps1 * boundary_mean) / mass,
preserving the penalty and face-average convention of the stored reproduction.
An independent quadrature evaluates fixed networks in double precision;
the original training trajectories and checkpoint selection are unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

#: The architecture the stored states were trained with.
DEFAULT_WIDTH = 150
DEFAULT_DEPTH = 4
#: The penalty weights, which differ between the two potentials.
BOUNDARY_PENALTY = 2000.0
NORMALIZATION_PENALTY = {"square": 1000.0, "exp": 500.0}


class SquaredReluResNet(nn.Module):
    r"""The residual network of Ji et al., with activation :math:`\max(0,t)^2`.

    The activation is continuously differentiable and piecewise quadratic.
    Spatial derivatives needed by the energy are obtained by automatic
    differentiation, with the framework convention at the kink.
    """

    def __init__(
        self, dimension: int, width: int = DEFAULT_WIDTH, depth: int = DEFAULT_DEPTH
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("the residual stack needs at least two layers")
        self.input = nn.Linear(dimension, width)
        self.blocks = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 1))
        self.output = nn.Linear(width, 1)
        self.reset_parameters()

    @staticmethod
    def activation(value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(value).square()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # A small output layer starts the iterate near zero, where the loss is
        # dominated by the normalization penalty rather than by the energy.
        nn.init.normal_(self.output.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.output.bias)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        state = self.activation(self.input(points))
        for block in self.blocks:
            state = state + self.activation(block(state))
        return self.output(state).squeeze(-1)


def potential(name: str, points: torch.Tensor) -> torch.Tensor:
    """The one-dimensional potential of the benchmark, on a tensor of points."""
    if name == "square":
        return points * points
    if name == "exp":
        return torch.exp(-math.pi * points)
    raise ValueError(f"unknown potential: {name}")


def sobol_boundary(
    count: int, dimension: int, seed: int, *, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    r"""Scrambled Sobol points on :math:`\partial(0,1)^d`, evenly over the faces.

    Each of the :math:`2d` faces receives the same number of points, and a point
    of a face is a Sobol point of the :math:`(d-1)`-dimensional cube with the
    missing coordinate set to zero or one.  Spreading them evenly rather than
    sampling the face at random is what keeps the penalty from being dominated by
    whichever face happened to be drawn most.
    """
    per_face = int(math.ceil(count / (2 * dimension)))
    engine = torch.quasirandom.SobolEngine(dimension - 1, scramble=True, seed=seed)
    base = engine.draw(per_face * 2 * dimension).to(dtype=dtype)
    points = torch.empty(per_face * 2 * dimension, dimension, dtype=dtype)
    row = 0
    for axis in range(dimension):
        for side in (0.0, 1.0):
            block = torch.empty(per_face, dimension, dtype=dtype)
            block[:, axis] = side
            block[:, [j for j in range(dimension) if j != axis]] = base[
                row : row + per_face
            ]
            points[row : row + per_face] = block
            row += per_face
    return points[:count]


@dataclass
class EnergyTerms:
    """The pieces of the functional, kept apart because they are reported apart.

    ``interior_rayleigh`` is the quotient of the interior energy alone;
    ``penalized_rayleigh`` adds the boundary penalty to the numerator.  The two
    differ by a few percent for the stored states, and which one is reported has
    to be stated rather than assumed -- see :func:`evaluate_state`.
    """

    domain_energy: float
    mass: float
    boundary_integral: float
    boundary_penalty: float

    @property
    def interior_rayleigh(self) -> float:
        return self.domain_energy / self.mass

    @property
    def penalized_rayleigh(self) -> float:
        return (
            self.domain_energy + self.boundary_penalty * self.boundary_integral
        ) / self.mass


def evaluate_state(
    model: nn.Module,
    name: str,
    scale: float,
    points: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    boundary_points: torch.Tensor | None = None,
    boundary_penalty: float = BOUNDARY_PENALTY,
    chunk: int = 4096,
) -> EnergyTerms:
    r"""Evaluate a fixed state on the supplied interior and boundary rules.

    Weighted interior integration and equal-face boundary averaging preserve
    the stored reproduction's functional. Both interior and penalized quotients
    are returned; the reported baseline value is the penalized quotient.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    total = points.shape[0]
    energy = 0.0
    mass = 0.0

    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        block = torch.as_tensor(points[start:stop], device=device, dtype=dtype)
        block = block.detach().requires_grad_(True)
        values = model(block)
        gradients = torch.autograd.grad(values.sum(), block, create_graph=False)[0]
        reaction = scale * potential(name, block).sum(dim=1)
        energy_density = (
            gradients.square().sum(dim=1) + reaction * values.square()
        ).detach()
        mass_density = values.square().detach()
        if weights is None:
            share = (stop - start) / total
            energy += share * float(energy_density.mean())
            mass += share * float(mass_density.mean())
        else:
            block_weights = torch.as_tensor(
                weights[start:stop], device=device, dtype=dtype
            )
            energy += float(torch.sum(block_weights * energy_density))
            mass += float(torch.sum(block_weights * mass_density))

    boundary_integral = 0.0
    if boundary_points is not None:
        with torch.no_grad():
            count = boundary_points.shape[0]
            for start in range(0, count, chunk):
                stop = min(start + chunk, count)
                block = boundary_points[start:stop].to(device=device, dtype=dtype)
                boundary_integral += ((stop - start) / count) * float(
                    model(block).square().mean()
                )

    if mass <= 0.0:
        raise RuntimeError("the stored state has nonpositive mass on this rule")
    return EnergyTerms(energy, mass, boundary_integral, boundary_penalty)


def load_checkpoint(
    path, dimension: int = 10, *, width: int = DEFAULT_WIDTH, depth: int = DEFAULT_DEPTH
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild a stored state in double precision, with its recorded metadata.

    The states were trained in single precision; promoting them here removes the
    evaluation's own precision from the comparison without changing the state.
    The metadata carries which epoch was selected and on what grounds, which the
    manuscript reports and which a reader should be able to check.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = SquaredReluResNet(dimension, width, depth)
    model.load_state_dict(
        {
            key: value.to(dtype=torch.float64)
            for key, value in payload["state_dict"].items()
        }
    )
    metadata = {
        key: payload[key]
        for key in ("kind", "mode", "selected_epoch", "selected_by")
        if key in payload
    }
    if "final_row" in payload:
        for key in ("epoch", "stop_reason", "orth_penalty", "stationarity_residual"):
            if key in payload["final_row"]:
                metadata[key] = payload["final_row"][key]
    return model.to(dtype=torch.float64).eval(), metadata
