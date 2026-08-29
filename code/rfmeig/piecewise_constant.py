r"""Exact assembly of cosine features against coefficients constant on a cell grid.

Example 4 has coefficients that jump by a factor of thirty, but they jump only
on an axis-aligned grid of equal square cells.  Every entry of the pencil is
therefore a sum of cell integrals over which the coefficients are constant, and a
cell integral of a product of cosines has a closed form.  No quadrature enters the
assembly at all: the discrete problem is the exact Galerkin projection of the
continuous one onto the random feature space, and the only error left is the
approximation error of that space.

Two routes to the same matrices are given.

:func:`assemble_by_cells`
    the direct statement -- loop over cells, add the closed-form contribution --
    which is the definition and is what the fast route is checked against;

:func:`assemble`
    the same sum, rearranged so that the cell loop disappears.

The rearrangement is worth stating, because it is the difference between an
assembly that fits in memory and one that does not.  Writing a cell integral in
midpoint form,

.. math::

    \int_{m-h/2}^{m+h/2}\cos(f t)\,dt = h\,\mathrm{sinc}(fh/2)\cos(fm),

the sinc factor depends on the frequency pair but not on the cell, so it is one
fixed matrix per axis; and the cosine factor expands by the angle-addition
formula into a rank-two matrix per cell, so a cell-weighted sum of the seventeen
blocks has rank at most thirty-four and is one product with a narrow factor.  The
coefficient maps are seventeen by seventeen and are exactly reproduced by nine or
fewer singular vectors, so each double sum over cells collapses to that many such
products.  The direct route forms sixty-eight full matrices of the feature count
squared -- nineteen gigabytes at six thousand features -- and this one forms none.

The midpoint form also matters for accuracy.  The obvious way to write the cell
integral divides by the frequency difference, which is nearly zero for the
closest of six thousand random pairs; the form above never divides by it.
"""

from __future__ import annotations

import numpy as np

#: Coefficient maps below this fraction of their largest singular value are
#: treated as exhausted; the maps here are reproduced well above it.
SINGULAR_TOLERANCE = 1.0e-12


class AxisKernel:
    """Everything about one axis that does not depend on the cell weights.

    Built once per axis and reused by every coefficient map and every boundary
    term, which is where most of the saving comes from.
    """

    def __init__(self, frequencies: np.ndarray, cells: int, cell_size: float):
        self.frequencies = frequencies
        self.cells = cells
        midpoints = cell_size * (np.arange(cells) + 0.5)
        angles = midpoints[:, None] * frequencies[None, :]
        self.midpoint_sine = np.sin(angles)
        self.midpoint_cosine = np.cos(angles)

        # The one half of the product-to-sum identity is folded in here, so the
        # block routines need no extra pass over a full matrix.
        half = 0.5 * cell_size
        self.difference_sinc = half * np.sinc(
            half * (frequencies[:, None] - frequencies[None, :]) / np.pi
        )
        self.total_sinc = half * np.sinc(
            half * (frequencies[:, None] + frequencies[None, :]) / np.pi
        )

    def _weighted_midpoint_cosines(
        self, weights: np.ndarray, sign: float
    ) -> np.ndarray:
        r""":math:`\sum_c w_c\cos((f_i\mp f_j)m_c)`, by the angle-addition formula.

        The expansion into a cosine and a sine part is what makes the sum a
        product of two narrow matrices instead of a loop over cells.
        """
        count = self.frequencies.size
        left = np.empty((count, 2 * self.cells))
        right = np.empty_like(left)
        left[:, 0::2] = self.midpoint_cosine.T * weights[None, :]
        left[:, 1::2] = sign * self.midpoint_sine.T * weights[None, :]
        right[:, 0::2] = self.midpoint_cosine.T
        right[:, 1::2] = self.midpoint_sine.T
        return left @ right.T

    def cosine_and_sine_blocks(
        self, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""The weighted sums of :math:`\int\cos\cos` and :math:`\int\sin\sin`."""
        difference = self._weighted_midpoint_cosines(weights, 1.0)
        difference *= self.difference_sinc
        total = self._weighted_midpoint_cosines(weights, -1.0)
        total *= self.total_sinc
        cosine = difference + total
        np.subtract(difference, total, out=difference)
        return cosine, difference

    def cosine_block(self, weights: np.ndarray) -> np.ndarray:
        r"""The weighted sum of :math:`\int\cos\cos` alone."""
        difference = self._weighted_midpoint_cosines(weights, 1.0)
        difference *= self.difference_sinc
        total = self._weighted_midpoint_cosines(weights, -1.0)
        total *= self.total_sinc
        difference += total
        return difference


def exact_factors(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A factorization ``matrix = left @ right.T`` of the smallest width.

    Exact, not approximate: the singular values that are dropped are zero to
    working precision, and the tolerance above is checked against the largest.
    """
    left, singular, right = np.linalg.svd(matrix)
    keep = singular > SINGULAR_TOLERANCE * singular[0]
    return left[:, keep] * singular[keep][None, :], right[keep].T


def _cell_cosine_integral(
    frequency: np.ndarray, phase: np.ndarray, left: float, right: float
) -> np.ndarray:
    r""":math:`\int_l^r\cos(ft+p)\,dt`, with the zero-frequency case separated."""
    output = np.empty_like(frequency)
    nonzero = np.abs(frequency) > 1.0e-14
    output[nonzero] = (
        np.sin(frequency[nonzero] * right + phase[nonzero])
        - np.sin(frequency[nonzero] * left + phase[nonzero])
    ) / frequency[nonzero]
    output[~nonzero] = (right - left) * np.cos(phase[~nonzero])
    return output


def _cell_pair_integrals(
    frequencies: np.ndarray, phases: np.ndarray, left: float, right: float
) -> tuple[np.ndarray, np.ndarray]:
    r"""The cell integrals of :math:`\cos\cos` and of :math:`\sin\sin`, as matrices."""
    first = frequencies[:, None]
    second = frequencies[None, :]
    first_phase = phases[:, None]
    second_phase = phases[None, :]
    difference = _cell_cosine_integral(
        first - second, first_phase - second_phase, left, right
    )
    total = _cell_cosine_integral(
        first + second, first_phase + second_phase, left, right
    )
    return 0.5 * (difference + total), 0.5 * (difference - total)


def assemble_by_cells(
    frequencies_x: np.ndarray,
    frequencies_y: np.ndarray,
    diffusion: np.ndarray,
    absorption: np.ndarray,
    fission_map: np.ndarray,
    boundary_edges,
    cell_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The definition: loop over cells and boundary faces, adding closed forms.

    ``diffusion``, ``absorption`` and ``fission_map`` are indexed ``[row, column]``
    and are zero outside the domain.  Kept because it is the readable statement of
    what is being assembled and because :func:`assemble` is checked against it;
    it is not the route to use at the sizes the experiment runs.
    """
    cells = diffusion.shape[0]
    count = frequencies_x.size
    phases = np.zeros(count)

    cosine_x, sine_x, cosine_y, sine_y = [], [], [], []
    for index in range(cells):
        left = cell_size * index
        pair = _cell_pair_integrals(frequencies_x, phases, left, left + cell_size)
        cosine_x.append(pair[0])
        sine_x.append(pair[1])
        pair = _cell_pair_integrals(frequencies_y, phases, left, left + cell_size)
        cosine_y.append(pair[0])
        sine_y.append(pair[1])

    pair_x = frequencies_x[:, None] * frequencies_x[None, :]
    pair_y = frequencies_y[:, None] * frequencies_y[None, :]
    energy = np.zeros((count, count))
    fission = np.zeros_like(energy)

    for row, column in np.argwhere(absorption + fission_map + diffusion > 0.0):
        mass = cosine_x[column] * cosine_y[row]
        energy += diffusion[row, column] * (
            pair_x * sine_x[column] * cosine_y[row]
            + pair_y * cosine_x[column] * sine_y[row]
        )
        energy += absorption[row, column] * mass
        fission += fission_map[row, column] * mass

    for column, row, dx, dy in boundary_edges:
        if dx:
            coordinate = cell_size * (column + (dx > 0))
            trace = np.cos(frequencies_x * coordinate)
            energy += 0.5 * trace[:, None] * trace[None, :] * cosine_y[row]
        else:
            coordinate = cell_size * (row + (dy > 0))
            trace = np.cos(frequencies_y * coordinate)
            energy += 0.5 * trace[:, None] * trace[None, :] * cosine_x[column]

    return 0.5 * (energy + energy.T), 0.5 * (fission + fission.T)


def assemble(
    frequencies_x: np.ndarray,
    frequencies_y: np.ndarray,
    diffusion: np.ndarray,
    absorption: np.ndarray,
    fission_map: np.ndarray,
    boundary_edges,
    cell_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The same matrices, with the cell loop replaced by narrow products.

    The coefficient maps are transposed to ``[column, row]`` here because the
    factorization is taken across the two axes, and each factor then weights one
    axis kernel.
    """
    cells = diffusion.shape[0]
    count = frequencies_x.size
    x_axis = AxisKernel(frequencies_x, cells, cell_size)
    y_axis = AxisKernel(frequencies_y, cells, cell_size)

    energy = np.zeros((count, count))
    fission = np.zeros((count, count))
    scratch = np.empty((count, count))

    # The two coefficients that multiply the function itself.
    for coefficient_map, target in (
        (absorption.T, energy),
        (fission_map.T, fission),
    ):
        left, right = exact_factors(coefficient_map)
        for column in range(left.shape[1]):
            np.multiply(
                x_axis.cosine_block(left[:, column]),
                y_axis.cosine_block(right[:, column]),
                out=scratch,
            )
            target += scratch

    # The coefficient that multiplies the gradient, one term per direction.
    pair_x = frequencies_x[:, None] * frequencies_x[None, :]
    pair_y = frequencies_y[:, None] * frequencies_y[None, :]
    left, right = exact_factors(diffusion.T)
    for column in range(left.shape[1]):
        x_cosine, x_sine = x_axis.cosine_and_sine_blocks(left[:, column])
        y_cosine, y_sine = y_axis.cosine_and_sine_blocks(right[:, column])
        np.multiply(x_sine, y_cosine, out=scratch)
        scratch *= pair_x
        energy += scratch
        np.multiply(x_cosine, y_sine, out=scratch)
        scratch *= pair_y
        energy += scratch

    # The vacuum boundary: one symmetric update per boundary row or column,
    # because every face in a row shares the same transverse cell integral.
    traces_by_row: dict[int, list[np.ndarray]] = {}
    traces_by_column: dict[int, list[np.ndarray]] = {}
    for column, row, dx, dy in boundary_edges:
        if dx:
            coordinate = cell_size * (column + (dx > 0))
            traces_by_row.setdefault(row, []).append(np.cos(frequencies_x * coordinate))
        else:
            coordinate = cell_size * (row + (dy > 0))
            traces_by_column.setdefault(column, []).append(
                np.cos(frequencies_y * coordinate)
            )

    for row, traces in traces_by_row.items():
        weights = np.zeros(cells)
        weights[row] = 1.0
        transverse = y_axis.cosine_block(weights)
        block = np.stack(traces, axis=1)
        np.matmul(block, block.T, out=scratch)
        scratch *= transverse
        energy += 0.5 * scratch

    for column, traces in traces_by_column.items():
        weights = np.zeros(cells)
        weights[column] = 1.0
        transverse = x_axis.cosine_block(weights)
        block = np.stack(traces, axis=1)
        np.matmul(block, block.T, out=scratch)
        scratch *= transverse
        energy += 0.5 * scratch

    energy += energy.T
    energy *= 0.5
    fission += fission.T
    fission *= 0.5
    return energy, fission
