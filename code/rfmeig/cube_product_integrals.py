"""Product integrals for independent evaluation of fixed cube cosine fields.

The functions are eta(x) cos(omega.x + phase), eta=prod(4*x*(1-x)).
This evaluator uses no eigenfunctions or eigenvalues of the reference problem.
It does not replace the full-dimensional sampled matrix assembly.
"""

from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn


def _moment(frequency, polynomial):
    """Integral on [0,1] of P(2x-1) exp(i frequency x)."""
    legendre = np.polynomial.legendre.poly2leg(polynomial)
    frequency = np.asarray(frequency)
    real_argument = not np.iscomplexobj(frequency)
    result = np.zeros(frequency.shape, dtype=complex)
    for degree, coefficient in enumerate(legendre):
        if coefficient:
            if real_argument:
                bessel = spherical_jn(degree, np.abs(frequency) / 2)
                if degree % 2:
                    bessel = np.where(frequency < 0, -bessel, bessel)
            else:
                bessel = spherical_jn(degree, frequency / 2)
            result += coefficient * (1j**degree) * bessel
    return np.exp(0.5j * frequency) * result


def cosine_product_pencil(omega, phase, *, potential="square", scale=20.0):
    """Independently integrate the exact mass and energy of the given atoms."""
    omega = np.asarray(omega, dtype=float)
    phase = np.asarray(phase, dtype=float)
    count, dimension = omega.shape
    if phase.shape != (count,) or potential not in {"square", "exp"}:
        raise ValueError("invalid phases or potential")
    mass = np.zeros((count, count))
    energy = np.zeros_like(mass)
    squared = np.sum(omega * omega, axis=1)
    # f=1-t^2, f'= -4t, x=(1+t)/2.
    f_squared = np.array([1.0, 0.0, -2.0, 0.0, 1.0])
    square_weighted = np.polynomial.polynomial.polymul(f_squared, [0.25, 0.5, 0.25])
    for sign in (-1, 1):
        frequency = omega[:, None, :] + sign * omega[None, :, :]
        angle = phase[:, None] + sign * phase[None, :]
        moments = [_moment(frequency[:, :, axis], f_squared) for axis in range(dimension)]
        prefix = [np.ones_like(moments[0])]
        for moment in moments:
            prefix.append(prefix[-1] * moment)
        suffix = np.ones_like(moments[0])
        gradient = np.zeros_like(suffix)
        reaction = np.zeros_like(suffix)
        for axis in range(dimension - 1, -1, -1):
            rest = prefix[axis] * suffix
            s = frequency[:, :, axis]
            gradient += _moment(s, [0.0, 0.0, 16.0]) * rest
            reaction += (
                _moment(s, square_weighted)
                if potential == "square"
                else _moment(s + 1j * np.pi, f_squared)
            ) * rest
            suffix *= moments[axis]
        phase_factor = np.exp(1j * angle)
        part_mass = 0.5 * np.real(phase_factor * prefix[-1])
        mass += part_mass
        energy += 0.5 * np.real(phase_factor * (gradient + scale * reaction))
        energy += 0.5 * (squared[:, None] + squared[None, :]) * part_mass
    return 0.5 * (energy + energy.T), 0.5 * (mass + mass.T)


def fixed_cube_observables(omega, phase, coefficients, *, potential="square", scale=20.0):
    energy, mass = cosine_product_pencil(omega, phase, potential=potential, scale=scale)
    reduced_mass = coefficients.T @ mass @ coefficients
    reduced_energy = coefficients.T @ energy @ coefficients
    masses = np.diag(reduced_mass)
    correlation = reduced_mass / np.sqrt(masses[:, None] * masses[None, :])
    return {
        "rayleigh": np.diag(reduced_energy) / masses,
        "mass": masses,
        "orthogonality_max": float(np.max(np.abs(correlation - np.eye(len(masses))))),
        "mass_matrix": reduced_mass,
        "energy_matrix": reduced_energy,
    }
