r"""Device-resident twins of the assemblies and solves, for same-device timing.

Three of the six experiments compare the random feature method against a neural
baseline that was trained on a graphics card.  A wall time measured across two
devices is not a comparison of algorithms in either direction, so either the
baselines move to the host -- weeks of retraining -- or the method moves to the
card.  This module is that move.

Nothing here changes a discretization.  Every routine performs the same
operations, in the same order, as its counterpart in :mod:`rfmeig.rayleigh_ritz`
or :mod:`rfmeig.piecewise_constant`; only the array library differs, so the two
paths agree to summation order and the tests assert exactly that.  What does
change is memory management: an eight-gigabyte card cannot hold the temporaries a
host implementation is free to leave to the garbage collector, so the assemblies
below keep a fixed number of arrays alive and release each temporary as it goes.

One consequence of "to summation order" deserves stating, because it is larger
than it sounds.  The matrices agree to about 1e-15, but the reported Ritz value
can differ by 1e-6, and the amplifier is the truncation: a whitened direction
sitting near the threshold is kept by one path and dropped by the other, and one
direction more or fewer moves the value by far more than the matrices differ.
That is a property of the discretization and not of the device -- two host
assemblies of the same problem do it too -- and it stays well below the
discretization error being reported, but a device result should be compared with
a host one at that level rather than at round-off.

Putting the assembly here rather than only the solve matters for the same reason
the module exists at all.  With assembly left on the host, most of an Example 4
"GPU" solve was still host work, and calling the result a device measurement
would repeat in one experiment the mistake the module is meant to remove from
three.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from rfmeig.piecewise_constant import exact_factors


def available() -> bool:
    """Whether a card is present, without importing torch when it is not needed."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def require(device: str):
    """Resolve a device name, refusing to fall back silently.

    A silent fallback to the host is the one behaviour this module must not
    have: it would report a host time under a device label, which is the error
    the whole module exists to prevent.
    """
    import torch

    target = torch.device(device)
    if target.type != "cuda":
        return target
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{device} was requested but no CUDA device is available; the "
            "device is always explicit here and never falls back to the host"
        )
    count = torch.cuda.device_count()
    if target.index is not None and target.index >= count:
        raise RuntimeError(
            f"{device} was requested but only {count} CUDA device(s) are "
            "present; the device is always explicit here and never falls back "
            "to the host"
        )
    return target


def _synchronize(target) -> None:
    """Wait for the queue, so that a timing covers the work and not the launch."""
    import torch

    if target.type == "cuda":
        torch.cuda.synchronize(target)


def peak_memory_gib(target) -> float:
    """Largest device allocation since the last reset, in gibibytes."""
    import torch

    if target.type != "cuda":
        return float("nan")
    return float(torch.cuda.max_memory_allocated() / 2**30)


# --------------------------------------------------------------------------
# Example 2: the blocked cosine assembly
# --------------------------------------------------------------------------
def cosine_pencil(
    points: np.ndarray,
    weights: np.ndarray,
    omega: np.ndarray,
    phase: np.ndarray,
    factor,
    *,
    diffusion: np.ndarray | float = 1.0,
    reaction: np.ndarray | float = 0.0,
    block: int = 8192,
    device: str = "cuda",
):
    """The device twin of :func:`rfmeig.rayleigh_ritz.assemble_cosine_pencil`.

    Same blocking, same per-direction accumulation, and the subtraction that
    defines the feature gradient stays in the same place -- which is what keeps
    the cancellation behaviour, and therefore the reported digits, identical.
    """
    import torch

    target = require(device)
    dtype = torch.float64
    dimension = points.shape[1]
    count = omega.shape[0]

    def to_device(array):
        # A copy, not a view: a broadcast array is not writable and torch
        # will not take ownership of one.
        return torch.as_tensor(
            np.ascontiguousarray(array, dtype=np.float64).copy(),
            device=target,
            dtype=dtype,
        )

    points_t = to_device(points)
    weights_t = to_device(weights)
    omega_t = to_device(omega)
    phase_t = to_device(phase)
    eta, grad_eta = factor(points)
    eta_t = to_device(eta)
    grad_eta_t = to_device(grad_eta)
    diffusion_t = to_device(
        np.broadcast_to(np.asarray(diffusion, float), weights.shape)
    )
    reaction_t = to_device(
        np.broadcast_to(np.asarray(reaction, float), weights.shape)
    )

    def weighted_gram(matrix, scale):
        """The device has no symmetric rank-k update, so a general product is
        taken and symmetrized.  Multiplying by the weight rather than by its
        square root also keeps the routine correct for a signed weight, which a
        quadrature rule may carry."""
        gram = matrix.T @ (matrix * scale[:, None])
        return 0.5 * (gram + gram.T)

    eta_squared = eta_t * eta_t
    root_weight = torch.sqrt(weights_t * diffusion_t)
    mass = torch.zeros((count, count), device=target, dtype=dtype)
    reaction_matrix = torch.zeros_like(mass)
    energy = torch.zeros_like(mass)

    for start in range(0, points.shape[0], block):
        stop = min(start + block, points.shape[0])
        argument = points_t[start:stop] @ omega_t.T
        argument += phase_t[None, :]
        cosine = torch.cos(argument)
        sine = torch.sin(argument)
        del argument

        block_weights = weights_t[start:stop]
        block_eta_squared = eta_squared[start:stop]
        mass += weighted_gram(cosine, block_weights * block_eta_squared)
        reaction_matrix += weighted_gram(
            cosine, block_weights * reaction_t[start:stop] * block_eta_squared
        )

        scaled_sine = sine * (root_weight[start:stop] * eta_t[start:stop])[:, None]
        scaled_cosine = cosine * root_weight[start:stop][:, None]
        del cosine, sine
        for axis in range(dimension):
            column = scaled_cosine * grad_eta_t[start:stop, axis][:, None]
            column -= scaled_sine * omega_t[None, :, axis]
            gram = column.T @ column
            energy += 0.5 * (gram + gram.T)
            del column, gram
        del scaled_sine, scaled_cosine

    energy += reaction_matrix
    del reaction_matrix
    _synchronize(target)
    return energy, mass


# --------------------------------------------------------------------------
# Example 4: the rank-structured assembly on cell-constant coefficients
# --------------------------------------------------------------------------
class _AxisKernel:
    """Device twin of :class:`rfmeig.piecewise_constant.AxisKernel`.

    The midpoint tables are stored transposed, because every use below wants
    them that way and a transpose of a resident array is free only if it is not
    then fed to a matrix product.
    """

    def __init__(self, frequencies, cells: int, cell_size: float):
        import torch

        self.frequencies = frequencies
        self.cells = cells
        device, dtype = frequencies.device, frequencies.dtype
        midpoints = cell_size * (
            torch.arange(cells, device=device, dtype=dtype) + 0.5
        )
        angles = midpoints[:, None] * frequencies[None, :]
        self.midpoint_sine_t = torch.sin(angles).T.contiguous()
        self.midpoint_cosine_t = torch.cos(angles).T.contiguous()

        half = 0.5 * cell_size
        self.difference_sinc = half * torch.sinc(
            half * (frequencies[:, None] - frequencies[None, :]) / torch.pi
        )
        self.total_sinc = half * torch.sinc(
            half * (frequencies[:, None] + frequencies[None, :]) / torch.pi
        )

    def _weighted_midpoint_cosines(self, weights, sign: float):
        import torch

        count = self.frequencies.shape[0]
        device, dtype = self.frequencies.device, self.frequencies.dtype
        left = torch.empty((count, 2 * self.cells), device=device, dtype=dtype)
        right = torch.empty_like(left)
        left[:, 0::2] = self.midpoint_cosine_t * weights[None, :]
        left[:, 1::2] = sign * self.midpoint_sine_t * weights[None, :]
        right[:, 0::2] = self.midpoint_cosine_t
        right[:, 1::2] = self.midpoint_sine_t
        return left @ right.T

    def cosine_and_sine_blocks(self, weights):
        difference = self._weighted_midpoint_cosines(weights, 1.0)
        difference *= self.difference_sinc
        total = self._weighted_midpoint_cosines(weights, -1.0)
        total *= self.total_sinc
        cosine = difference + total
        difference -= total
        del total
        return cosine, difference

    def cosine_block(self, weights):
        difference = self._weighted_midpoint_cosines(weights, 1.0)
        difference *= self.difference_sinc
        total = self._weighted_midpoint_cosines(weights, -1.0)
        total *= self.total_sinc
        difference += total
        del total
        return difference


def piecewise_constant_pencil(
    frequencies_x: np.ndarray,
    frequencies_y: np.ndarray,
    diffusion: np.ndarray,
    absorption: np.ndarray,
    fission_map: np.ndarray,
    boundary_edges,
    cell_size: float,
    *,
    device: str = "cuda",
):
    """The device twin of :func:`rfmeig.piecewise_constant.assemble`.

    The host routine's saving was in never forming the per-cell tables; the
    saving here is on top of that, and it is of a different kind.  What is left
    after the rearrangement is a handful of thin matrix products and several
    passes over square arrays -- the first compute-bound, the second
    bandwidth-bound, and a consumer card is good at both even in double
    precision.  Three square arrays stay resident and every temporary is
    released, because at six thousand features each is 288 mebibytes and the
    eigendecomposition that follows needs nearly two gibibytes of its own.
    """
    import torch

    target = require(device)
    dtype = torch.float64
    cells = diffusion.shape[0]
    count = frequencies_x.shape[0]

    def to_device(array):
        return torch.as_tensor(
            np.ascontiguousarray(array, dtype=np.float64).copy(),
            device=target,
            dtype=dtype,
        )

    alpha = to_device(frequencies_x)
    beta = to_device(frequencies_y)
    x_axis = _AxisKernel(alpha, cells, cell_size)
    y_axis = _AxisKernel(beta, cells, cell_size)

    energy = torch.zeros((count, count), device=target, dtype=dtype)
    fission = torch.zeros((count, count), device=target, dtype=dtype)

    def factors(matrix: np.ndarray):
        left, right = exact_factors(matrix)
        return to_device(left), to_device(right)

    for coefficient_map, out in ((absorption.T, energy), (fission_map.T, fission)):
        left, right = factors(coefficient_map)
        for column in range(left.shape[1]):
            block = x_axis.cosine_block(left[:, column])
            block *= y_axis.cosine_block(right[:, column])
            out += block
            del block

    pair_x = alpha[:, None] * alpha[None, :]
    pair_y = beta[:, None] * beta[None, :]
    left, right = factors(diffusion.T)
    for column in range(left.shape[1]):
        x_cosine, x_sine = x_axis.cosine_and_sine_blocks(left[:, column])
        y_cosine, y_sine = y_axis.cosine_and_sine_blocks(right[:, column])
        x_sine *= y_cosine
        x_sine *= pair_x
        energy += x_sine
        x_cosine *= y_sine
        x_cosine *= pair_y
        energy += x_cosine
        del x_cosine, x_sine, y_cosine, y_sine
    del pair_x, pair_y

    traces_by_row: dict[int, list] = {}
    traces_by_column: dict[int, list] = {}
    for column, row, dx, dy in boundary_edges:
        if dx:
            coordinate = cell_size * (column + (dx > 0))
            traces_by_row.setdefault(row, []).append(torch.cos(alpha * coordinate))
        else:
            coordinate = cell_size * (row + (dy > 0))
            traces_by_column.setdefault(column, []).append(torch.cos(beta * coordinate))

    for lookup, axis in ((traces_by_row, y_axis), (traces_by_column, x_axis)):
        for index, traces in lookup.items():
            weights = torch.zeros(cells, device=target, dtype=dtype)
            weights[index] = 1.0
            block = torch.stack(traces, dim=1)
            update = block @ block.T
            update *= axis.cosine_block(weights)
            energy += 0.5 * update
            del block, update

    energy += energy.T.clone()
    energy *= 0.5
    fission += fission.T.clone()
    fission *= 0.5
    _synchronize(target)
    return energy, fission


# --------------------------------------------------------------------------
# the solves
# --------------------------------------------------------------------------
def dominant_generalized_eigenpair(
    fission,
    whitening,
    *,
    iterations: int = 5000,
    tolerance: float = 1.0e-15,
):
    r"""The largest eigenpair of :math:`W^\top F W`, without forming that matrix.

    Forming it would cost two products of a six-thousand by five-thousand matrix
    and a full tridiagonalization, to read off one number.  Only the dominant
    pair is wanted, the operator needs three matrix--vector products per step,
    and the criticality problem's fundamental mode is well separated, so plain
    power iteration converges quickly.

    **The stopping test is on the eigenvalue, but the eigenvector is what has to
    converge.**  Power iteration drives the Rayleigh quotient to its limit
    quadratically faster than the vector, so a value that looks settled to twelve
    digits can sit on a vector whose residual is only :math:`10^{-7}`.  That
    matters here because the reported flux error is a maximum-norm functional of
    the vector and not of the value: at a value tolerance of :math:`10^{-13}` the
    vector residual was :math:`4.5\times10^{-7}` and the flux error moved in its
    fourth digit.  At :math:`10^{-15}` the residual reaches :math:`1.3\times
    10^{-9}`, which is the floor this iteration attains on this operator;
    stopping on the residual directly buys nothing and costs six times the steps.
    The residual actually achieved is returned rather than assumed.
    """
    import torch

    vector = torch.ones(
        whitening.shape[1], dtype=whitening.dtype, device=whitening.device
    )
    vector /= torch.linalg.vector_norm(vector)
    value = torch.zeros((), dtype=whitening.dtype, device=whitening.device)
    used = iterations
    for step in range(iterations):
        product = whitening.T @ (fission @ (whitening @ vector))
        norm = torch.linalg.vector_norm(product)
        if norm == 0:
            break
        product = product / norm
        previous, value = value, norm
        vector = product
        if torch.abs(value - previous) <= tolerance * torch.abs(value):
            used = step + 1
            break

    applied = whitening.T @ (fission @ (whitening @ vector))
    rayleigh = vector @ applied
    residual = float(
        torch.linalg.vector_norm(applied - rayleigh * vector) / torch.abs(rayleigh)
    )
    return float(rayleigh), vector, residual, used


def solve_energy_whitened_dominant(
    energy,
    fission,
    *,
    tolerance: float,
    device: str = "cuda",
) -> dict[str, Any]:
    r"""Whiten by the energy form and take the dominant fission mode.

    The route Example 4 uses on the device: the same threshold on the same
    scaled spectrum as the host path, then the power iteration above instead of a
    reduced eigendecomposition.  The multiplication factor is read directly --
    it *is* the dominant eigenvalue -- rather than as the reciprocal of a small
    one.
    """
    import torch

    target = require(device)
    if not isinstance(energy, torch.Tensor):
        energy = torch.as_tensor(energy, device=target, dtype=torch.float64)
        fission = torch.as_tensor(fission, device=target, dtype=torch.float64)

    started = time.perf_counter()
    values, vectors = torch.linalg.eigh(energy)
    threshold = max(float(values[-1]) * tolerance, 1.0e-15)
    retained = values > threshold
    whitening = vectors[:, retained] / torch.sqrt(values[retained])[None, :]
    del vectors
    _synchronize(target)
    whiten_seconds = time.perf_counter() - started

    started = time.perf_counter()
    keff, reduced, residual, iterations = dominant_generalized_eigenpair(
        fission, whitening
    )
    coefficients = (whitening @ reduced).cpu().numpy()
    _synchronize(target)
    dominant_seconds = time.perf_counter() - started

    return {
        "keff": keff,
        "coefficients": coefficients,
        "retained": int(retained.sum().item()),
        "spectrum": (values / values[-1]).cpu().numpy(),
        "power_iteration_residual": residual,
        "power_iterations": iterations,
        "whiten_s": whiten_seconds,
        "dominant_s": dominant_seconds,
    }


def solve_mass_whitened(
    energy,
    mass,
    count: int,
    *,
    tolerance: float,
    device: str = "cuda",
) -> dict[str, Any]:
    """The device twin of :func:`rfmeig.rayleigh_ritz.solve_mass_whitened`.

    The same rescaling by the mass diagonal, the same threshold on the same
    scaled spectrum, the same whitening, and the same lowest Ritz values, in the
    same order, so the two paths differ only by summation order.  The relative
    residual of every returned pair is measured against the *original* pencil,
    after the whitening has been undone.
    """
    import torch

    target = require(device)
    energy_t = torch.as_tensor(energy, device=target, dtype=torch.float64)
    mass_t = torch.as_tensor(mass, device=target, dtype=torch.float64)
    original_energy, original_mass = energy_t, mass_t

    diagonal = torch.diagonal(mass_t).clone()
    positive = diagonal > 1e-30
    if not bool(positive.any()):
        raise RuntimeError("the mass matrix has no positive diagonal entry")
    energy_t = energy_t[positive][:, positive]
    mass_t = mass_t[positive][:, positive]

    scale = 1.0 / torch.sqrt(diagonal[positive])
    mass_scaled = (scale[:, None] * mass_t) * scale[None, :]
    energy_scaled = (scale[:, None] * energy_t) * scale[None, :]
    mass_scaled = 0.5 * (mass_scaled + mass_scaled.T)
    energy_scaled = 0.5 * (energy_scaled + energy_scaled.T)

    spectrum, directions = torch.linalg.eigh(mass_scaled)
    keep = spectrum > tolerance * spectrum[-1]
    if int(keep.sum().item()) < count:
        raise RuntimeError("the truncation threshold retained too few directions")
    whitening = directions[:, keep] / torch.sqrt(spectrum[keep])[None, :]
    reduced = whitening.T @ energy_scaled @ whitening
    reduced = 0.5 * (reduced + reduced.T)
    all_values, all_vectors = torch.linalg.eigh(reduced)
    values = all_values[:count]

    coefficients = torch.zeros(
        (positive.shape[0], count), device=target, dtype=torch.float64
    )
    coefficients[positive] = scale[:, None] * (whitening @ all_vectors[:, :count])

    applied_energy = original_energy @ coefficients
    applied_mass = original_mass @ coefficients
    residuals = torch.linalg.vector_norm(
        applied_energy - applied_mass * values[None, :], dim=0
    ) / torch.clamp_min(
        torch.linalg.vector_norm(applied_energy, dim=0)
        + torch.abs(values) * torch.linalg.vector_norm(applied_mass, dim=0),
        1.0e-30,
    )
    _synchronize(target)
    return {
        "eigenvalues": values.cpu().numpy(),
        "coefficients": coefficients.cpu().numpy(),
        "retained": int(keep.sum().item()),
        "spectrum": (spectrum / spectrum[-1]).cpu().numpy(),
        "residuals": residuals.cpu().numpy(),
    }


def cosine_pencil_expanded(
    points: np.ndarray,
    weights: np.ndarray,
    omega: np.ndarray,
    phase: np.ndarray,
    factor,
    *,
    reaction: np.ndarray | float = 0.0,
    block: int = 4096,
    device: str = "cuda",
):
    r"""The device twin of
    :func:`rfmeig.rayleigh_ritz.assemble_cosine_pencil_expanded`.

    Example 3 assembles this way because the cost stops growing with the
    dimension, and it is the form the recorded ten-dimensional run used.  The
    same caveat applies here as on the host: the cancellation between the terms
    costs about :math:`\log_{10}|\omega|^2` digits, which is under two at that
    experiment's frequency scale and would be four or five at Example 2's.

    At the sizes of Example 3 the pencil itself is a few megabytes, so nothing
    here is near the memory budget; what the device buys is evaluating the
    features at two hundred thousand points, which is the whole cost.
    """
    import torch

    target = require(device)
    dtype = torch.float64
    count = omega.shape[0]

    def to_device(array):
        return torch.as_tensor(
            np.ascontiguousarray(array, dtype=np.float64).copy(),
            device=target,
            dtype=dtype,
        )

    points_t = to_device(points)
    weights_t = to_device(weights)
    omega_t = to_device(omega)
    phase_t = to_device(phase)
    reaction_t = to_device(np.broadcast_to(np.asarray(reaction, float), weights.shape))
    frequency_gram = omega_t @ omega_t.T

    mass = torch.zeros((count, count), device=target, dtype=dtype)
    energy = torch.zeros_like(mass)
    for start in range(0, points.shape[0], block):
        stop = min(start + block, points.shape[0])
        eta, grad_eta = factor(points[start:stop])
        eta_t = to_device(eta)
        grad_eta_t = to_device(grad_eta)

        argument = points_t[start:stop] @ omega_t.T + phase_t
        cosine = torch.cos(argument)
        sine = torch.sin(argument)
        del argument

        block_weights = weights_t[start:stop]
        eta_squared = eta_t * eta_t
        grad_squared = torch.sum(grad_eta_t * grad_eta_t, dim=1)

        mass += cosine.T @ ((block_weights * eta_squared)[:, None] * cosine)
        energy += cosine.T @ (
            (block_weights * (grad_squared + reaction_t[start:stop] * eta_squared))[
                :, None
            ]
            * cosine
        )
        cross = cosine.T @ (
            ((block_weights * eta_t)[:, None] * sine) * (grad_eta_t @ omega_t.T)
        )
        energy -= cross + cross.T
        energy += (
            sine.T @ ((block_weights * eta_squared)[:, None] * sine)
        ) * frequency_gram
        del cosine, sine, cross

    energy = 0.5 * (energy + energy.T)
    mass = 0.5 * (mass + mass.T)
    _synchronize(target)
    return energy, mass
