r"""Deep Ritz on the ten-dimensional cube: the baseline of Example 3.

This is the method of Ji et al. (2024), and it differs from the Deep Ritz of
Example 2 in both of the ways that matter.  The network is a residual stack
with a squared rectified activation rather than a hyperbolic-tangent one, and the
boundary condition is imposed by a penalty rather than by multiplication:

.. math::

    \mathcal{L}[u]=\tfrac12\Big(\int_\Omega|\nabla u|^2+\gamma V u^2
        +\varepsilon_1\int_{\partial\Omega}u^2\Big)
        +\tfrac{\varepsilon_2}{4}\Big((\|u\|^2-1)^2
        +2\sum_{k<m}\langle u,u_k\rangle^2\Big).

The penalty is why the reported quantity has to be defined with care.  The
iterate is not in :math:`H_0^1(\Omega)`, so the quotient is not an upper bound
for the eigenvalue and the boundary term is not part of it; what is reported is
the interior Rayleigh quotient of the trained state.  Higher modes are found by
adding the orthogonality penalty against the modes already accepted, which is
the deflation the last sum performs.

The three modes of each potential were trained on a graphics card in single
precision, and their states are stored in the repository.  What this module
supplies is the network, the functional, and the evaluation of a stored state on
an arbitrary rule -- in particular on the rule the random feature method
assembles with, which is what makes the two columns of Table 3 comparable.  The
trajectories themselves are not retrained here.
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

    A squared rectifier is twice differentiable where a plain one is not, which
    the energy needs: the loss contains :math:`|\nabla u|^2` and is itself
    differentiated, so the activation has to survive two derivatives.  The
    residual connections are what let the depth be increased without the early
    layers stopping to train.
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
    r"""The energy, the mass and the boundary term of a fixed state on a rule.

    ``weights`` makes the interior estimate a quadrature; without them the points
    carry equal weight, which is the Monte Carlo estimate the trajectory itself
    was validated with.  Supplying the weights of the rule the random feature
    method assembles with is what puts both methods of Table 3 on one integration
    rule, and it is the whole reason this function takes a rule at all rather
    than sampling its own.

    Which quotient to report is a real choice, not a detail.  The iterate is not
    zero on the boundary, so the interior quotient alone is not the quantity the
    method minimizes and is systematically *below* the eigenvalue -- by about
    three percent for the stored states.  What the method minimizes, and what the
    source reports, is the penalized quotient, in which the boundary term is
    charged at :math:`\varepsilon_1`.  Table 3 reports the penalized one, so that
    the baseline is quoted the way its own authors quote it; both are returned
    here and the difference is recorded with every run.
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
