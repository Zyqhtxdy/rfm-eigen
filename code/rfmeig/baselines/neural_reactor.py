r"""The three neural baselines of Example 4, and what separates them.

All three train the same network on the same collocation points and differ only
in the objective:

``drm``
    Deep Ritz.  Minimizes the Rayleigh quotient of the weak form, with the
    vacuum condition entering as the natural boundary term it is.  It never sees
    the strong equation, so the jump in the diffusion coefficient never appears
    explicitly -- and the network, being globally smooth, cannot represent the
    kink the jump produces.

``gipmnn``
    The inverse power method as a network.  Minimizes the residual of the strong
    equation against a source that is frozen at each step from the previous
    iterate, plus the residual of the boundary conditions.  Freezing the source
    is what makes the problem at each step linear, and it is the neural
    counterpart of one inverse iteration.

``pc_gipmnn``
    The same, with the network given one output per subdomain -- four control
    rods, two fuels, the reflector -- and with the continuity of the flux and of
    the normal current imposed at every interface between two of them.  This is
    the only one of the three that can represent the kink at all, because it is
    the only one that is not a single smooth function across the interface.

The flux is the square of the network output, which keeps it positive without a
constraint; that is the source paper's device and it is kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from rfmeig.problems.iaea_reactor import (
    MATERIALS,
    SUBDOMAIN_MATERIAL,
    active_area,
    outer_length,
)

METHODS = ("drm", "gipmnn", "pc_gipmnn")
#: ``pc_gipmnn`` needs one output per subdomain; the other two need one.
OUTPUTS = {"drm": 1, "gipmnn": 1, "pc_gipmnn": 7}


class ResidualFluxNetwork(nn.Module):
    """The two-block hyperbolic-tangent residual network of the source paper.

    The input is rescaled to roughly :math:`[-1,1]` before the first layer, which
    a network on a domain a hundred and seventy centimetres wide needs if its
    initialization is to mean anything.
    """

    def __init__(self, width: int = 20, outputs: int = 1, extent: float = 85.0):
        super().__init__()
        self.extent = float(extent)
        self.first = nn.Linear(2, width)
        self.second = nn.Linear(width, width)
        self.first_skip = nn.Linear(2, width)
        self.third = nn.Linear(width, width)
        self.fourth = nn.Linear(width, width)
        self.second_skip = nn.Linear(width, width)
        self.output = nn.Linear(width, outputs)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        scaled = points / self.extent - 1.0
        block = torch.tanh(self.second(torch.tanh(self.first(scaled)))) + torch.tanh(
            self.first_skip(scaled)
        )
        block = torch.tanh(self.fourth(torch.tanh(self.third(block)))) + torch.tanh(
            self.second_skip(block)
        )
        return self.output(block)

    def flux(
        self, points: torch.Tensor, subdomains: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The flux: the selected output, squared so that it stays positive."""
        raw = self(points)
        if subdomains is not None:
            raw = raw.gather(1, subdomains[:, None]).squeeze(1)
        elif raw.shape[1] == 1:
            raw = raw[:, 0]
        else:
            raise ValueError("a multi-output network needs subdomain indices")
        return raw.square()


# --------------------------------------------------------------------------
# coefficients and derivatives
# --------------------------------------------------------------------------
def material_tensors(
    labels: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three cross sections at points carrying the given material labels."""
    diffusion = torch.zeros(labels.shape, device=labels.device, dtype=dtype)
    absorption = torch.zeros_like(diffusion)
    fission = torch.zeros_like(diffusion)
    for key, material in MATERIALS.items():
        mask = labels == key
        diffusion[mask] = material.diffusion
        absorption[mask] = material.absorption
        fission[mask] = material.fission
    return diffusion, absorption, fission


def flux_and_gradient(
    model: ResidualFluxNetwork,
    points: torch.Tensor,
    subdomains: torch.Tensor | None,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The flux and its gradient, with the points returned so they can be reused."""
    coordinates = points.detach().requires_grad_(True)
    flux = model.flux(coordinates, subdomains)
    gradient = torch.autograd.grad(flux.sum(), coordinates, create_graph=create_graph)[
        0
    ]
    return coordinates, flux, gradient


def strong_operator(
    model: ResidualFluxNetwork,
    points: torch.Tensor,
    labels: torch.Tensor,
    subdomains: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r""":math:`-D\Delta\phi+\Sigma_a\phi`, and the fission cross section beside it.

    The Laplacian is taken one direction at a time through a second application
    of automatic differentiation, which is why the objectives that use it cost
    several times what the weak one does.
    """
    coordinates, flux, gradient = flux_and_gradient(
        model, points, subdomains, create_graph=True
    )
    laplacian = torch.zeros_like(flux)
    for axis in range(2):
        second = torch.autograd.grad(
            gradient[:, axis].sum(), coordinates, create_graph=True
        )[0][:, axis]
        laplacian = laplacian + second
    diffusion, absorption, fission = material_tensors(labels, flux.dtype)
    return flux, -diffusion * laplacian + absorption * flux, fission


def inverse_iteration_source(
    flux: torch.Tensor, applied: torch.Tensor, fission: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The Rayleigh value of the current iterate, and the source frozen from it.

    Both are detached: freezing the source is what makes the step linear, and a
    gradient through it would undo that.
    """
    denominator = torch.clamp_min((fission * flux.square()).sum(), 1.0e-30)
    eigenvalue = (applied * flux).sum() / denominator
    mean_power = torch.clamp_min(flux.mean(), 1.0e-30)
    return eigenvalue.detach(), (flux / mean_power).detach()


# --------------------------------------------------------------------------
# the three objectives
# --------------------------------------------------------------------------
def weak_rayleigh(
    model: ResidualFluxNetwork,
    points: torch.Tensor,
    labels: torch.Tensor,
    subdomains: torch.Tensor | None,
    outer_points: torch.Tensor,
    outer_subdomains: torch.Tensor | None,
) -> torch.Tensor:
    r"""The Deep Ritz objective: the weak Rayleigh quotient, with the vacuum term.

    The boundary integral appears in the numerator with the weight the vacuum
    condition gives it, so it is part of the energy rather than a penalty; the
    two integrals are means over their point sets, multiplied by the measure each
    set covers.
    """
    _, flux, gradient = flux_and_gradient(model, points, subdomains, create_graph=True)
    diffusion, absorption, fission = material_tensors(labels, flux.dtype)
    volume = (
        diffusion * gradient.square().sum(dim=1) + absorption * flux.square()
    ).mean()
    production = (fission * flux.square()).mean()
    outer_flux = model.flux(outer_points, outer_subdomains)
    numerator = (
        active_area() * volume + outer_length() * 0.5 * outer_flux.square().mean()
    )
    return numerator / torch.clamp_min(active_area() * production, 1.0e-30)


def boundary_residual(
    model: ResidualFluxNetwork,
    symmetry_points: torch.Tensor,
    symmetry_normals: torch.Tensor,
    symmetry_labels: torch.Tensor,
    symmetry_subdomains: torch.Tensor | None,
    outer_points: torch.Tensor,
    outer_normals: torch.Tensor,
    outer_labels: torch.Tensor,
    outer_subdomains: torch.Tensor | None,
) -> torch.Tensor:
    r"""Both boundary conditions, as squared residuals of the normal current.

    On the symmetry axes the current must vanish; on the outer boundary it must
    balance a quarter of the flux, which is the vacuum condition.
    """
    _, _, symmetry_gradient = flux_and_gradient(
        model, symmetry_points, symmetry_subdomains, create_graph=True
    )
    symmetry_diffusion, _, _ = material_tensors(
        symmetry_labels, symmetry_gradient.dtype
    )
    symmetry_current = (
        -0.5 * symmetry_diffusion * (symmetry_gradient * symmetry_normals).sum(dim=1)
    )

    _, outer_flux, outer_gradient = flux_and_gradient(
        model, outer_points, outer_subdomains, create_graph=True
    )
    outer_diffusion, _, _ = material_tensors(outer_labels, outer_flux.dtype)
    outer_current = -0.5 * outer_diffusion * (outer_gradient * outer_normals).sum(dim=1)
    return (
        symmetry_current.square().mean()
        + (outer_current - 0.25 * outer_flux).square().mean()
    )


def interface_residual(
    model: ResidualFluxNetwork,
    points: torch.Tensor,
    normals: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    left_labels: torch.Tensor,
    right_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The two interface conditions: the flux and the normal current must match.

    Only the subdomain-aware method has anything to impose here; for the other
    two the flux is one function and both conditions hold trivially -- which is
    exactly why they cannot represent the kink.
    """
    coordinates = points.detach().requires_grad_(True)
    raw = model(coordinates)
    left_flux = raw.gather(1, left[:, None]).squeeze(1).square()
    right_flux = raw.gather(1, right[:, None]).squeeze(1).square()
    left_gradient = torch.autograd.grad(
        left_flux.sum(), coordinates, create_graph=True
    )[0]
    right_gradient = torch.autograd.grad(
        right_flux.sum(), coordinates, create_graph=True
    )[0]
    left_diffusion, _, _ = material_tensors(left_labels, left_flux.dtype)
    right_diffusion, _, _ = material_tensors(right_labels, right_flux.dtype)
    left_current = -left_diffusion * (left_gradient * normals).sum(dim=1)
    right_current = -right_diffusion * (right_gradient * normals).sum(dim=1)
    return (
        (left_flux - right_flux).square().mean(),
        (left_current - right_current).square().mean(),
    )


@dataclass
class TensorCollocation:
    """The collocation set, moved onto the device once and reused every epoch."""

    points: torch.Tensor
    labels: torch.Tensor
    subdomains: torch.Tensor | None
    symmetry_points: torch.Tensor
    symmetry_normals: torch.Tensor
    symmetry_labels: torch.Tensor
    symmetry_subdomains: torch.Tensor | None
    outer_points: torch.Tensor
    outer_normals: torch.Tensor
    outer_labels: torch.Tensor
    outer_subdomains: torch.Tensor | None
    interface_points: torch.Tensor
    interface_normals: torch.Tensor
    interface_left: torch.Tensor
    interface_right: torch.Tensor
    interface_left_labels: torch.Tensor
    interface_right_labels: torch.Tensor

    @classmethod
    def from_collocation(
        cls, data, method: str, device: torch.device, dtype: torch.dtype
    ) -> TensorCollocation:
        def real(array) -> torch.Tensor:
            return torch.as_tensor(np.asarray(array), device=device, dtype=dtype)

        def whole(array) -> torch.Tensor:
            return torch.as_tensor(np.asarray(array), device=device, dtype=torch.int64)

        use_subdomains = OUTPUTS[method] > 1
        # The material on each side of an interface follows from the subdomain
        # label alone, so it is looked up rather than resampled from the geometry.
        left_material = SUBDOMAIN_MATERIAL[np.asarray(data.interface_left)]
        right_material = SUBDOMAIN_MATERIAL[np.asarray(data.interface_right)]
        return cls(
            points=real(data.interior),
            labels=whole(data.material),
            subdomains=whole(data.subdomain) if use_subdomains else None,
            symmetry_points=real(data.symmetry_points),
            symmetry_normals=real(data.symmetry_normals),
            symmetry_labels=whole(data.symmetry_material),
            symmetry_subdomains=whole(data.symmetry_subdomain)
            if use_subdomains
            else None,
            outer_points=real(data.outer_points),
            outer_normals=real(data.outer_normals),
            outer_labels=whole(data.outer_material),
            outer_subdomains=whole(data.outer_subdomain) if use_subdomains else None,
            interface_points=real(data.interface_points),
            interface_normals=real(data.interface_normals),
            interface_left=whole(data.interface_left),
            interface_right=whole(data.interface_right),
            interface_left_labels=whole(left_material),
            interface_right_labels=whole(right_material),
        )


def objective(
    method: str, model: ResidualFluxNetwork, data: TensorCollocation
) -> dict[str, torch.Tensor]:
    """The loss of one method, with its parts kept apart for the record.

    The residual terms are multiplied by the number of points in their set, which
    is how the source paper weights them: it turns each mean back into a sum, so
    that a set with more points counts for more.
    """
    zero = torch.zeros((), device=data.points.device, dtype=data.points.dtype)
    if method == "drm":
        value = weak_rayleigh(
            model,
            data.points,
            data.labels,
            data.subdomains,
            data.outer_points,
            data.outer_subdomains,
        )
        return {"loss": value, "equation": value, "boundary": zero, "interface": zero}

    flux, applied, fission = strong_operator(
        model, data.points, data.labels, data.subdomains
    )
    eigenvalue, source = inverse_iteration_source(flux, applied, fission)
    equation = (
        data.points.shape[0] * (applied - eigenvalue * fission * source).square().mean()
    )
    boundary = data.symmetry_points.shape[0] * boundary_residual(
        model,
        data.symmetry_points,
        data.symmetry_normals,
        data.symmetry_labels,
        data.symmetry_subdomains,
        data.outer_points,
        data.outer_normals,
        data.outer_labels,
        data.outer_subdomains,
    )
    interface = zero
    if method == "pc_gipmnn":
        continuity, current = interface_residual(
            model,
            data.interface_points,
            data.interface_normals,
            data.interface_left,
            data.interface_right,
            data.interface_left_labels,
            data.interface_right_labels,
        )
        interface = data.interface_points.shape[0] * (continuity + current)
    return {
        "loss": equation + boundary + interface,
        "equation": equation,
        "boundary": boundary,
        "interface": interface,
    }


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def evaluate_keff(
    model: ResidualFluxNetwork, data: TensorCollocation, chunk: int = 4096
) -> float:
    r"""The multiplication factor of a trained state, from the weak quotient.

    The same functional the Deep Ritz objective minimizes, evaluated on a fixed
    state, so that all three methods are read off in one way regardless of what
    each of them was trained on.  The reciprocal at the end is because the
    quotient is the loss and the reported quantity is
    :math:`k_{\mathrm{eff}}=1/\mathcal{R}`.
    """
    from rfmeig.problems.iaea_reactor import material_fields

    model.eval()
    numerator = 0.0
    denominator = 0.0
    total = data.points.shape[0]
    for start in range(0, total, chunk):
        stop = min(total, start + chunk)
        selected = data.subdomains[start:stop] if data.subdomains is not None else None
        coordinates = data.points[start:stop].detach().requires_grad_(True)
        flux = model.flux(coordinates, selected)
        gradient = torch.autograd.grad(flux.sum(), coordinates, create_graph=False)[0]
        diffusion, absorption, fission = (
            torch.as_tensor(field, device=data.points.device, dtype=data.points.dtype)
            for field in material_fields(data.labels[start:stop].detach().cpu().numpy())
        )
        share = (stop - start) / total
        numerator += share * float(
            (diffusion * gradient.square().sum(dim=1) + absorption * flux.square())
            .mean()
            .detach()
            .cpu()
        )
        denominator += share * float((fission * flux.square()).mean().detach().cpu())
    with torch.no_grad():
        outer_mean = float(
            model.flux(data.outer_points, data.outer_subdomains)
            .square()
            .mean()
            .detach()
            .cpu()
        )
    quotient = (active_area() * numerator + outer_length() * 0.5 * outer_mean) / (
        active_area() * denominator
    )
    return 1.0 / quotient


def evaluate_flux(
    model: ResidualFluxNetwork,
    points: torch.Tensor,
    subdomains: torch.Tensor | None,
    chunk: int = 8192,
) -> np.ndarray:
    """The flux of a trained state at the given points, on the host."""
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, points.shape[0], chunk):
            stop = min(points.shape[0], start + chunk)
            selected = subdomains[start:stop] if subdomains is not None else None
            values.append(
                model.flux(points[start:stop], selected).detach().cpu().numpy()
            )
    return np.concatenate(values)
