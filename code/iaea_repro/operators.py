from __future__ import annotations

import torch

from .geometry import MATERIALS, active_area, robin_length
from .networks import ResidualFluxNet


def inverse_iteration_state(
    flux: torch.Tensor,
    applied: torch.Tensor,
    nu_sigma_f: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the Rayleigh value and mean-power normalized source."""
    denominator = torch.clamp_min((nu_sigma_f * flux.square()).sum(), 1.0e-30)
    eigenvalue = (applied * flux).sum() / denominator
    mean_power = torch.clamp_min(flux.mean(), 1.0e-30)
    return eigenvalue.detach(), (flux / mean_power).detach()


def material_tensors(labels: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    diffusion = torch.zeros(labels.shape, device=labels.device, dtype=dtype)
    sigma_a = torch.zeros_like(diffusion)
    nu_sigma_f = torch.zeros_like(diffusion)
    for key, material in MATERIALS.items():
        mask = labels == key
        diffusion[mask] = material.diffusion
        sigma_a[mask] = material.sigma_a
        nu_sigma_f[mask] = material.nu_sigma_f
    return diffusion, sigma_a, nu_sigma_f


def flux_and_gradient(
    model: ResidualFluxNet,
    points: torch.Tensor,
    subdomains: torch.Tensor | None,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coordinates = points.detach().requires_grad_(True)
    flux = model.flux(coordinates, subdomains)
    gradient = torch.autograd.grad(flux.sum(), coordinates, create_graph=create_graph)[0]
    return coordinates, flux, gradient


def strong_operator(
    model: ResidualFluxNet,
    points: torch.Tensor,
    labels: torch.Tensor,
    subdomains: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coordinates, flux, gradient = flux_and_gradient(model, points, subdomains, create_graph=True)
    laplacian = torch.zeros_like(flux)
    for axis in range(2):
        second = torch.autograd.grad(gradient[:, axis].sum(), coordinates, create_graph=True)[0][:, axis]
        laplacian = laplacian + second
    diffusion, sigma_a, nu_sigma_f = material_tensors(labels, flux.dtype)
    applied = -diffusion * laplacian + sigma_a * flux
    return flux, applied, nu_sigma_f


def weak_rayleigh(
    model: ResidualFluxNet,
    points: torch.Tensor,
    labels: torch.Tensor,
    subdomains: torch.Tensor | None,
    robin_points: torch.Tensor,
    robin_subdomains: torch.Tensor | None,
) -> torch.Tensor:
    _, flux, gradient = flux_and_gradient(model, points, subdomains, create_graph=True)
    diffusion, sigma_a, nu_sigma_f = material_tensors(labels, flux.dtype)
    volume_energy = (diffusion * gradient.square().sum(dim=1) + sigma_a * flux.square()).mean()
    fission = (nu_sigma_f * flux.square()).mean()
    robin_flux = model.flux(robin_points, robin_subdomains)
    numerator = active_area() * volume_energy + robin_length() * 0.5 * robin_flux.square().mean()
    denominator = active_area() * fission
    return numerator / torch.clamp_min(denominator, 1.0e-30)


def boundary_loss(
    model: ResidualFluxNet,
    symmetry_points: torch.Tensor,
    symmetry_normals: torch.Tensor,
    symmetry_labels: torch.Tensor,
    symmetry_subdomains: torch.Tensor | None,
    robin_points: torch.Tensor,
    robin_normals: torch.Tensor,
    robin_labels: torch.Tensor,
    robin_subdomains: torch.Tensor | None,
) -> torch.Tensor:
    _, symmetry_flux, symmetry_gradient = flux_and_gradient(
        model, symmetry_points, symmetry_subdomains, create_graph=True
    )
    symmetry_diffusion, _, _ = material_tensors(symmetry_labels, symmetry_flux.dtype)
    symmetry_current = -0.5 * symmetry_diffusion * (symmetry_gradient * symmetry_normals).sum(dim=1)

    _, robin_flux, robin_gradient = flux_and_gradient(model, robin_points, robin_subdomains, create_graph=True)
    robin_diffusion, _, _ = material_tensors(robin_labels, robin_flux.dtype)
    robin_current = -0.5 * robin_diffusion * (robin_gradient * robin_normals).sum(dim=1)
    robin_residual = robin_current - 0.25 * robin_flux
    return symmetry_current.square().mean() + robin_residual.square().mean()


def interface_loss(
    model: ResidualFluxNet,
    points: torch.Tensor,
    normals: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    left_material: torch.Tensor,
    right_material: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = points.detach().requires_grad_(True)
    raw = model(coordinates)
    left_flux = raw.gather(1, left[:, None]).squeeze(1).square()
    right_flux = raw.gather(1, right[:, None]).squeeze(1).square()
    left_gradient = torch.autograd.grad(left_flux.sum(), coordinates, create_graph=True)[0]
    right_gradient = torch.autograd.grad(right_flux.sum(), coordinates, create_graph=True)[0]
    left_diffusion, _, _ = material_tensors(left_material, left_flux.dtype)
    right_diffusion, _, _ = material_tensors(right_material, right_flux.dtype)
    left_current = -left_diffusion * (left_gradient * normals).sum(dim=1)
    right_current = -right_diffusion * (right_gradient * normals).sum(dim=1)
    return (left_flux - right_flux).square().mean(), (left_current - right_current).square().mean()
