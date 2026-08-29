from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

DOMAIN_SIZE = 170.0
CELL_SIZE = 10.0
GEOMETRY_TOP_TO_BOTTOM = (
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


def geometry_sha256() -> str:
    payload = ("\n".join(GEOMETRY_TOP_TO_BOTTOM) + "\n").encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Material:
    sigma_a: float
    sigma_s: float
    nu_sigma_f: float

    @property
    def diffusion(self) -> float:
        return 1.0 / (3.0 * (self.sigma_a + self.sigma_s))


MATERIALS = {
    1: Material(2.0, 0.53, 0.079),
    2: Material(0.087, 0.55, 0.085),
    3: Material(0.38, 0.20, 0.0),
    4: Material(0.01, 0.89, 0.0),
}


def cell_labels() -> np.ndarray:
    rows = [[0 if char == "." else int(char) for char in row] for row in GEOMETRY_TOP_TO_BOTTOM]
    return np.asarray(rows[::-1], dtype=np.int8)


def cell_subdomains() -> np.ndarray:
    materials = cell_labels()
    components, count = ndimage.label(materials == 3)
    if count != 4:
        raise RuntimeError(f"Expected four control-rod components, found {count}")
    order: list[tuple[float, float, int]] = []
    for component in range(1, count + 1):
        iy, ix = np.where(components == component)
        order.append((float(iy.mean()), float(ix.mean()), component))
    order.sort()
    subdomains = np.full_like(materials, -1, dtype=np.int8)
    for output_id, (_, _, component) in enumerate(order):
        subdomains[components == component] = output_id
    subdomains[materials == 2] = 4
    subdomains[materials == 1] = 5
    subdomains[materials == 4] = 6
    return subdomains


def labels_at(points: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    labels = cell_labels() if labels is None else labels
    clipped = np.clip(points, 0.0, np.nextafter(DOMAIN_SIZE, 0.0))
    ix = np.floor(clipped[:, 0] / CELL_SIZE).astype(int)
    iy = np.floor(clipped[:, 1] / CELL_SIZE).astype(int)
    out = labels[iy, ix].copy()
    outside = np.any((points < 0.0) | (points > DOMAIN_SIZE), axis=1)
    out[outside] = 0 if labels.min() >= 0 else -1
    return out


def material_arrays(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diffusion = np.zeros(labels.shape, dtype=np.float64)
    sigma_a = np.zeros(labels.shape, dtype=np.float64)
    nu_sigma_f = np.zeros(labels.shape, dtype=np.float64)
    for key, material in MATERIALS.items():
        mask = labels == key
        diffusion[mask] = material.diffusion
        sigma_a[mask] = material.sigma_a
        nu_sigma_f[mask] = material.nu_sigma_f
    return diffusion, sigma_a, nu_sigma_f


@dataclass(frozen=True)
class CollocationData:
    interior: np.ndarray
    material: np.ndarray
    subdomain: np.ndarray
    symmetry_points: np.ndarray
    symmetry_normals: np.ndarray
    symmetry_material: np.ndarray
    symmetry_subdomain: np.ndarray
    robin_points: np.ndarray
    robin_normals: np.ndarray
    robin_material: np.ndarray
    robin_subdomain: np.ndarray
    interface_points: np.ndarray
    interface_normals: np.ndarray
    interface_left: np.ndarray
    interface_right: np.ndarray


def _sample_edge(start: tuple[float, float], end: tuple[float, float], count: int) -> np.ndarray:
    t = (np.arange(count, dtype=float) + 0.5) / count
    start_array = np.asarray(start, dtype=float)
    return start_array[None, :] + t[:, None] * (np.asarray(end, dtype=float) - start_array)[None, :]


def build_collocation(nx: int = 171, points_per_cell: int = 10) -> CollocationData:
    coordinates = np.linspace(0.0, DOMAIN_SIZE, nx)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    material = labels_at(grid)
    active = material > 0
    interior = grid[active]
    interior_material = material[active]
    subdomain_map = cell_subdomains()
    interior_subdomain = labels_at(interior, subdomain_map)

    materials = cell_labels()
    symmetry_points: list[np.ndarray] = []
    symmetry_normals: list[np.ndarray] = []
    symmetry_material: list[np.ndarray] = []
    symmetry_subdomain: list[np.ndarray] = []
    robin_points: list[np.ndarray] = []
    robin_normals: list[np.ndarray] = []
    robin_material: list[np.ndarray] = []
    robin_subdomain: list[np.ndarray] = []
    interface_points: list[np.ndarray] = []
    interface_normals: list[np.ndarray] = []
    interface_left: list[np.ndarray] = []
    interface_right: list[np.ndarray] = []
    directions = (
        (1, 0, np.array([1.0, 0.0])),
        (-1, 0, np.array([-1.0, 0.0])),
        (0, 1, np.array([0.0, 1.0])),
        (0, -1, np.array([0.0, -1.0])),
    )

    for iy in range(17):
        for ix in range(17):
            if materials[iy, ix] <= 0:
                continue
            x0, y0 = CELL_SIZE * ix, CELL_SIZE * iy
            for dx, dy, normal in directions:
                jx, jy = ix + dx, iy + dy
                neighbor_active = 0 <= jx < 17 and 0 <= jy < 17 and materials[jy, jx] > 0
                if dx == 1:
                    edge = _sample_edge((x0 + CELL_SIZE, y0), (x0 + CELL_SIZE, y0 + CELL_SIZE), points_per_cell)
                elif dx == -1:
                    edge = _sample_edge((x0, y0), (x0, y0 + CELL_SIZE), points_per_cell)
                elif dy == 1:
                    edge = _sample_edge((x0, y0 + CELL_SIZE), (x0 + CELL_SIZE, y0 + CELL_SIZE), points_per_cell)
                else:
                    edge = _sample_edge((x0, y0), (x0 + CELL_SIZE, y0), points_per_cell)

                if not neighbor_active:
                    owner_material = np.full(edge.shape[0], materials[iy, ix], dtype=np.int64)
                    owner_subdomain = np.full(edge.shape[0], subdomain_map[iy, ix], dtype=np.int64)
                    if ix == 0 and dx == -1 or iy == 0 and dy == -1:
                        symmetry_points.append(edge)
                        symmetry_normals.append(np.repeat(normal[None, :], edge.shape[0], axis=0))
                        symmetry_material.append(owner_material)
                        symmetry_subdomain.append(owner_subdomain)
                    else:
                        robin_points.append(edge)
                        robin_normals.append(np.repeat(normal[None, :], edge.shape[0], axis=0))
                        robin_material.append(owner_material)
                        robin_subdomain.append(owner_subdomain)
                    continue
                if dx < 0 or dy < 0:
                    continue
                left = int(subdomain_map[iy, ix])
                right = int(subdomain_map[jy, jx])
                if left == right:
                    continue
                interface_points.append(edge)
                interface_normals.append(np.repeat(normal[None, :], edge.shape[0], axis=0))
                interface_left.append(np.full(edge.shape[0], left, dtype=np.int64))
                interface_right.append(np.full(edge.shape[0], right, dtype=np.int64))

    def stack(items: list[np.ndarray], width: int, dtype: np.dtype = np.float64) -> np.ndarray:
        return np.concatenate(items, axis=0).astype(dtype, copy=False) if items else np.empty((0, width), dtype=dtype)

    return CollocationData(
        interior=interior,
        material=interior_material.astype(np.int64),
        subdomain=interior_subdomain.astype(np.int64),
        symmetry_points=stack(symmetry_points, 2),
        symmetry_normals=stack(symmetry_normals, 2),
        symmetry_material=stack([item[:, None] for item in symmetry_material], 1, np.int64).ravel(),
        symmetry_subdomain=stack([item[:, None] for item in symmetry_subdomain], 1, np.int64).ravel(),
        robin_points=stack(robin_points, 2),
        robin_normals=stack(robin_normals, 2),
        robin_material=stack([item[:, None] for item in robin_material], 1, np.int64).ravel(),
        robin_subdomain=stack([item[:, None] for item in robin_subdomain], 1, np.int64).ravel(),
        interface_points=stack(interface_points, 2),
        interface_normals=stack(interface_normals, 2),
        interface_left=stack([item[:, None] for item in interface_left], 1, np.int64).ravel(),
        interface_right=stack([item[:, None] for item in interface_right], 1, np.int64).ravel(),
    )


def active_area() -> float:
    return float(np.count_nonzero(cell_labels()) * CELL_SIZE**2)


def robin_length() -> float:
    return float(build_collocation(nx=3, points_per_cell=1).robin_points.shape[0] * CELL_SIZE)
