r"""The truncated dipolar interaction symbol.

The dipolar potential is a convolution, and a convolution evaluated by a discrete
transform is periodic whether or not the problem is.  The symbol here is
truncated to the computational box, so that the transform returns the free-space
convolution over that box rather than the interaction of the condensate with its
own periodic images -- which at these box sizes would be of the same order as the
physical term.
"""
from __future__ import annotations

import numpy as np
import scipy.special
from numpy.typing import NDArray

RealArray = NDArray[np.float64]


def truncated_dipolar_symbol_2d(
    kx: RealArray,
    ky: RealArray,
    dipole_axis: tuple[float, float, float],
    truncation_radius: float,
) -> RealArray:
    """Fourier transform of the ball-truncated quasi-2D DDI kernel."""
    magnitude = np.sqrt(kx * kx + ky * ky)
    coulomb_transform = np.empty_like(magnitude)
    nonzero = magnitude > 0.0
    argument = truncation_radius * magnitude[nonzero]
    bessel_integral = 0.5 * np.pi * argument * (
        scipy.special.jv(0, argument) * scipy.special.struve(-1, argument)
        + scipy.special.jv(1, argument) * scipy.special.struve(0, argument)
    )
    coulomb_transform[nonzero] = (
        bessel_integral / magnitude[nonzero]
    )
    coulomb_transform[~nonzero] = truncation_radius

    n1, n2, n3 = dipole_axis
    directional = n1 * kx + n2 * ky
    return (
        1.5
        * (directional * directional - n3 * n3 * magnitude * magnitude)
        * coulomb_transform
    )
