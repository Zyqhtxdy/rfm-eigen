r"""Example 4: the IAEA two-dimensional benchmark, one energy group.

The problem is the criticality eigenvalue problem

.. math::

    -\nabla\cdot(D\nabla\phi)+\Sigma_a\phi
    =\frac{1}{k_{\mathrm{eff}}}\nu\Sigma_f\phi\quad\text{in }\Omega,

on a quarter core of a hundred and seventy centimetres a side, with reflective
conditions on the two symmetry axes and a vacuum condition
:math:`\tfrac12 D\partial_n\phi+\tfrac14\phi=0` on the outer boundary.  It is
posed on an eigenvalue that is *simple*, so it is not a test of multiplicity; it
is here because it is the benchmark the neural methods this section compares
against were built for, and because it is genuinely hard for a different reason.

The difficulty is the coefficients.  There are four materials, laid out on a
grid of square cells ten centimetres a side; across a cell face the absorption
cross section jumps by a factor of two hundred and the diffusion coefficient by
a factor of four.  What is continuous across such a face is the flux and the
normal current :math:`D\partial_n\phi`, so the normal derivative itself jumps by
the ratio of the two diffusion coefficients, and the solution is only
:math:`H^{3/2-\epsilon}` there.  A globally smooth trial function cannot
represent that kink.  What saves the method is that the jumps lie exactly on the
cell grid, so every integral splits into cell integrals over which the
coefficients are constant -- which is what :mod:`rfmeig.piecewise_constant`
exploits.

The reference is a cell-centred finite volume solution on a uniform refinement of
the same grid, with harmonic averaging of the diffusion across faces, which is
the standard discretization for exactly this kind of jump.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh

#: The side of the quarter core, in centimetres.
DOMAIN_SIZE = 170.0
#: The side of one material cell.  Every coefficient is constant inside a cell,
#: which is the structure the assembly is built around.
CELL_SIZE = 10.0
#: Cells per side.
CELLS = 17

#: The benchmark layout, written from the top row down as it is usually drawn.
#: ``.`` is outside the core.
LAYOUT_TOP_TO_BOTTOM = (
    "4444444..........",
    "4444444..........",
    "11111444444......",
    "11111444444......",
    "2221111114444....",
    "2221111114444....",
    "222222211114444..",
    "222222211114444..",
    "322222233111144..",
    "322222233111144..",
    "22222222222114444",
    "22222222222114444",
    "22222222222111144",
    "22222222222111144",
    "22222222222221144",
    "22222222222221144",
    "32222223322221144",
)


def layout_sha256() -> str:
    """A hash of the layout, so a run records which core it solved."""
    payload = ("\n".join(LAYOUT_TOP_TO_BOTTOM) + "\n").encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Material:
    """One material's cross sections, in inverse centimetres."""

    absorption: float
    scattering: float
    fission: float

    @property
    def diffusion(self) -> float:
        r""":math:`D=1/(3(\Sigma_a+\Sigma_s))`, the usual one-group relation."""
        return 1.0 / (3.0 * (self.absorption + self.scattering))


#: Fuel 1, fuel 2, the control rods, and the reflector.  The absorption cross
#: sections span a factor of two hundred and the diffusion coefficients a factor
#: of four; between them they are where the difficulty of the benchmark lies.
MATERIALS = {
    1: Material(2.0, 0.53, 0.079),
    2: Material(0.087, 0.55, 0.085),
    3: Material(0.38, 0.20, 0.0),
    4: Material(0.01, 0.89, 0.0),
}


def cell_materials() -> np.ndarray:
    """The material of every cell, indexed from the bottom row up."""
    rows = [
        [0 if char == "." else int(char) for char in row]
        for row in LAYOUT_TOP_TO_BOTTOM
    ]
    return np.asarray(rows[::-1], dtype=np.int8)


def cell_subdomains() -> np.ndarray:
    """A subdomain label per cell, which the neural baselines need.

    The rod-following methods give each connected control rod its own network
    output, so the four rods have to be told apart and ordered reproducibly --
    here by the position of their centres.  The three remaining materials take
    one label each.  A method that does not use subdomains ignores this.
    """
    materials = cell_materials()
    components, count = ndimage.label(materials == 3)
    if count != 4:
        raise RuntimeError(f"expected four control rods, found {count}")

    ordered = []
    for component in range(1, count + 1):
        rows, columns = np.where(components == component)
        ordered.append((float(rows.mean()), float(columns.mean()), component))
    ordered.sort()

    subdomains = np.full_like(materials, -1, dtype=np.int8)
    for label, (_, _, component) in enumerate(ordered):
        subdomains[components == component] = label
    subdomains[materials == 2] = 4
    subdomains[materials == 1] = 5
    subdomains[materials == 4] = 6
    return subdomains


#: The material each subdomain label is made of: four control rods, then the
#: second fuel, the first fuel and the reflector.  Fixed by ``cell_subdomains``.
SUBDOMAIN_MATERIAL = np.array([3, 3, 3, 3, 2, 1, 4], dtype=np.int64)


def labels_at(points: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    """The cell label containing each point; zero, or minus one, outside the core."""
    labels = cell_materials() if labels is None else labels
    clipped = np.clip(points, 0.0, np.nextafter(DOMAIN_SIZE, 0.0))
    column = np.floor(clipped[:, 0] / CELL_SIZE).astype(int)
    row = np.floor(clipped[:, 1] / CELL_SIZE).astype(int)
    found = labels[row, column].copy()
    outside = np.any((points < 0.0) | (points > DOMAIN_SIZE), axis=1)
    found[outside] = 0 if labels.min() >= 0 else -1
    return found


def material_fields(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three coefficient fields, sampled wherever the labels are given."""
    diffusion = np.zeros(labels.shape, dtype=np.float64)
    absorption = np.zeros(labels.shape, dtype=np.float64)
    fission = np.zeros(labels.shape, dtype=np.float64)
    for key, material in MATERIALS.items():
        mask = labels == key
        diffusion[mask] = material.diffusion
        absorption[mask] = material.absorption
        fission[mask] = material.fission
    return diffusion, absorption, fission


def active_area() -> float:
    """The area of the core, which the weak forms integrate over."""
    return float(np.count_nonzero(cell_materials()) * CELL_SIZE**2)


def outer_edges() -> list[tuple[int, int, int, int]]:
    """Every cell face on the vacuum boundary, as ``(column, row, dx, dy)``.

    A face is on the vacuum boundary when the neighbouring cell is outside the
    core and the face is not on one of the two symmetry axes, where the
    reflective condition applies instead and contributes nothing.
    """
    materials = cell_materials()
    edges: list[tuple[int, int, int, int]] = []
    for row in range(CELLS):
        for column in range(CELLS):
            if materials[row, column] <= 0:
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbour_column, neighbour_row = column + dx, row + dy
                inside = (
                    0 <= neighbour_column < CELLS
                    and 0 <= neighbour_row < CELLS
                    and materials[neighbour_row, neighbour_column] > 0
                )
                reflective = (column == 0 and dx == -1) or (row == 0 and dy == -1)
                if not inside and not reflective:
                    edges.append((column, row, dx, dy))
    return edges


def outer_length() -> float:
    """The length of the vacuum boundary."""
    return float(len(outer_edges()) * CELL_SIZE)


# --------------------------------------------------------------------------
# collocation points, which the neural baselines train on
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Collocation:
    """Interior, boundary and interface points, with the labels each carries."""

    interior: np.ndarray
    material: np.ndarray
    subdomain: np.ndarray
    symmetry_points: np.ndarray
    symmetry_normals: np.ndarray
    symmetry_material: np.ndarray
    symmetry_subdomain: np.ndarray
    outer_points: np.ndarray
    outer_normals: np.ndarray
    outer_material: np.ndarray
    outer_subdomain: np.ndarray
    interface_points: np.ndarray
    interface_normals: np.ndarray
    interface_left: np.ndarray
    interface_right: np.ndarray


def _edge_points(start, end, count: int) -> np.ndarray:
    """``count`` points evenly spaced along an edge, at cell midpoints."""
    parameter = (np.arange(count, dtype=float) + 0.5) / count
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    return start[None, :] + parameter[:, None] * (end - start)[None, :]


def build_collocation(grid: int = 171, per_edge: int = 10) -> Collocation:
    """The point set the neural baselines are trained and evaluated on.

    The interior is a uniform grid restricted to the core; the boundary points
    sit on the cell faces, separated into the reflective and the vacuum parts;
    and the interface points sit on every face between two different subdomains,
    where the continuity of the flux and of the normal current is imposed.
    """
    axis = np.linspace(0.0, DOMAIN_SIZE, grid)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    nodes = np.column_stack([x.ravel(), y.ravel()])
    material = labels_at(nodes)
    inside = material > 0
    subdomain_map = cell_subdomains()

    materials = cell_materials()
    symmetry: list[list] = [[], [], [], []]
    outer: list[list] = [[], [], [], []]
    interface: list[list] = [[], [], [], []]
    directions = (
        (1, 0, np.array([1.0, 0.0])),
        (-1, 0, np.array([-1.0, 0.0])),
        (0, 1, np.array([0.0, 1.0])),
        (0, -1, np.array([0.0, -1.0])),
    )

    for row in range(CELLS):
        for column in range(CELLS):
            if materials[row, column] <= 0:
                continue
            left, bottom = CELL_SIZE * column, CELL_SIZE * row
            for dx, dy, normal in directions:
                neighbour_column, neighbour_row = column + dx, row + dy
                inside_neighbour = (
                    0 <= neighbour_column < CELLS
                    and 0 <= neighbour_row < CELLS
                    and materials[neighbour_row, neighbour_column] > 0
                )
                if dx == 1:
                    edge = _edge_points(
                        (left + CELL_SIZE, bottom),
                        (left + CELL_SIZE, bottom + CELL_SIZE),
                        per_edge,
                    )
                elif dx == -1:
                    edge = _edge_points(
                        (left, bottom), (left, bottom + CELL_SIZE), per_edge
                    )
                elif dy == 1:
                    edge = _edge_points(
                        (left, bottom + CELL_SIZE),
                        (left + CELL_SIZE, bottom + CELL_SIZE),
                        per_edge,
                    )
                else:
                    edge = _edge_points(
                        (left, bottom), (left + CELL_SIZE, bottom), per_edge
                    )

                if not inside_neighbour:
                    owner_material = np.full(
                        edge.shape[0], materials[row, column], dtype=np.int64
                    )
                    owner_subdomain = np.full(
                        edge.shape[0], subdomain_map[row, column], dtype=np.int64
                    )
                    normals = np.repeat(normal[None, :], edge.shape[0], axis=0)
                    target = (
                        symmetry
                        if (column == 0 and dx == -1) or (row == 0 and dy == -1)
                        else outer
                    )
                    target[0].append(edge)
                    target[1].append(normals)
                    target[2].append(owner_material)
                    target[3].append(owner_subdomain)
                    continue

                # Each interior face is visited from both sides; keep one.
                if dx < 0 or dy < 0:
                    continue
                near = int(subdomain_map[row, column])
                far = int(subdomain_map[neighbour_row, neighbour_column])
                if near == far:
                    continue
                interface[0].append(edge)
                interface[1].append(np.repeat(normal[None, :], edge.shape[0], axis=0))
                interface[2].append(np.full(edge.shape[0], near, dtype=np.int64))
                interface[3].append(np.full(edge.shape[0], far, dtype=np.int64))

    def stack(parts: list[np.ndarray], width: int, dtype=np.float64) -> np.ndarray:
        if not parts:
            return np.empty((0, width) if width > 1 else (0,), dtype=dtype)
        return np.concatenate(parts, axis=0).astype(dtype, copy=False)

    return Collocation(
        interior=nodes[inside],
        material=material[inside].astype(np.int64),
        subdomain=labels_at(nodes[inside], subdomain_map).astype(np.int64),
        symmetry_points=stack(symmetry[0], 2),
        symmetry_normals=stack(symmetry[1], 2),
        symmetry_material=stack(symmetry[2], 1, np.int64),
        symmetry_subdomain=stack(symmetry[3], 1, np.int64),
        outer_points=stack(outer[0], 2),
        outer_normals=stack(outer[1], 2),
        outer_material=stack(outer[2], 1, np.int64),
        outer_subdomain=stack(outer[3], 1, np.int64),
        interface_points=stack(interface[0], 2),
        interface_normals=stack(interface[1], 2),
        interface_left=stack(interface[2], 1, np.int64),
        interface_right=stack(interface[3], 1, np.int64),
    )


# --------------------------------------------------------------------------
# the finite volume reference
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Reference:
    """A finite volume solution, with the interpolation the errors are taken on."""

    keff: float
    refinement: int
    labels: np.ndarray
    flux: np.ndarray

    def flux_at(self, points: np.ndarray) -> np.ndarray:
        """Bilinear interpolation of the cell-centred flux, normalized to a peak of one.

        Cells outside the core are excluded from each interpolation rather than
        treated as zero, so a point near the boundary is interpolated only from
        the cells that exist.
        """
        step = CELL_SIZE / self.refinement
        x_index = points[:, 0] / step - 0.5
        y_index = points[:, 1] / step - 0.5
        base_x = np.floor(x_index).astype(int)
        base_y = np.floor(y_index).astype(int)
        fraction_x = x_index - base_x
        fraction_y = y_index - base_y

        values = np.zeros(points.shape[0], dtype=float)
        weight_total = np.zeros(points.shape[0], dtype=float)
        for dx, weight_x in ((0, 1.0 - fraction_x), (1, fraction_x)):
            for dy, weight_y in ((0, 1.0 - fraction_y), (1, fraction_y)):
                column = np.clip(base_x + dx, 0, self.flux.shape[1] - 1)
                row = np.clip(base_y + dy, 0, self.flux.shape[0] - 1)
                weight = weight_x * weight_y
                valid = self.labels[row, column] > 0
                values += weight * valid * self.flux[row, column]
                weight_total += weight * valid
        values /= np.maximum(weight_total, 1.0e-30)
        return values / max(float(np.max(values)), 1.0e-30)


def solve_reference(refinement: int = 48, tolerance: float = 1.0e-10) -> Reference:
    r"""A cell-centred finite volume solution on a uniform refinement of the grid.

    The diffusion coefficient is averaged harmonically across each face, which is
    the averaging that makes the discrete normal current continuous and is the
    reason this discretization handles the jumps at all.  The vacuum condition
    contributes :math:`h/2` to the diagonal of a boundary cell and the reflective
    condition contributes nothing, as it should.

    The generalized problem is solved for the *largest* eigenvalue of the fission
    matrix against the loss matrix, which is :math:`k_{\mathrm{eff}}` directly and
    avoids inverting the loss operator.
    """
    labels = np.repeat(
        np.repeat(cell_materials(), refinement, axis=0), refinement, axis=1
    )
    step = CELL_SIZE / refinement
    inside = labels > 0
    index = -np.ones_like(labels, dtype=np.int64)
    index[inside] = np.arange(int(inside.sum()))
    diffusion, absorption, fission_field = material_fields(labels)

    unknowns = int(inside.sum())
    loss = lil_matrix((unknowns, unknowns), dtype=np.float64)
    fission = np.zeros(unknowns, dtype=np.float64)

    for row, column in np.argwhere(inside):
        here = int(index[row, column])
        loss[here, here] += absorption[row, column] * step * step
        fission[here] = fission_field[row, column] * step * step
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbour_column, neighbour_row = column + dx, row + dy
            in_core = (
                0 <= neighbour_column < labels.shape[1]
                and 0 <= neighbour_row < labels.shape[0]
                and inside[neighbour_row, neighbour_column]
            )
            if in_core:
                near = diffusion[row, column]
                far = diffusion[neighbour_row, neighbour_column]
                face = 2.0 * near * far / (near + far)
                loss[here, here] += face
                loss[here, int(index[neighbour_row, neighbour_column])] -= face
            elif not ((column == 0 and dx == -1) or (row == 0 and dy == -1)):
                loss[here, here] += 0.5 * step

    fission_matrix = csr_matrix(
        (fission, (np.arange(unknowns), np.arange(unknowns))),
        shape=(unknowns, unknowns),
    )
    values, vectors = eigsh(
        fission_matrix, M=loss.tocsr(), k=1, which="LA", tol=tolerance
    )
    flux = np.zeros_like(labels, dtype=np.float64)
    flux[inside] = vectors[:, 0]
    if float(np.mean(flux[inside])) < 0.0:
        flux = -flux
    flux /= float(np.max(flux))
    return Reference(float(values[0]), refinement, labels, flux)


# --------------------------------------------------------------------------
# the two reported errors
# --------------------------------------------------------------------------
def mean_power_normalize(values: np.ndarray) -> np.ndarray:
    """Divide by the discrete mean, which is how the source paper normalizes.

    An eigenfunction is defined only up to a constant, so a flux error is
    meaningless until the two fields are put on one scale.  This is the scaling
    the baseline's authors use, and it is used here so that their reported errors
    and these are the same quantity.
    """
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if abs(mean) <= 1.0e-30:
        raise ValueError("a flux with zero discrete mean cannot be normalized")
    return values / mean


def flux_error(approximation: np.ndarray, reference: np.ndarray) -> float:
    """The largest pointwise difference of the two normalized fluxes, relatively."""
    approximation = mean_power_normalize(approximation)
    reference = mean_power_normalize(reference)
    denominator = max(float(np.max(np.abs(reference))), 1.0e-30)
    return float(np.max(np.abs(approximation - reference)) / denominator)


def keff_error(keff: float, reference: float) -> float:
    """The relative error of the criticality eigenvalue."""
    return abs(float(keff) - float(reference)) / abs(float(reference))
