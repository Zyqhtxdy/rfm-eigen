from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh

from .geometry import CELL_SIZE, cell_labels, geometry_sha256, material_arrays


@dataclass(frozen=True)
class ReferenceSolution:
    keff: float
    subdiv: int
    labels: np.ndarray
    flux: np.ndarray

    def flux_at(self, points: np.ndarray) -> np.ndarray:
        h = CELL_SIZE / self.subdiv
        x_index = points[:, 0] / h - 0.5
        y_index = points[:, 1] / h - 0.5
        ix0 = np.floor(x_index).astype(int)
        iy0 = np.floor(y_index).astype(int)
        tx = x_index - ix0
        ty = y_index - iy0
        values = np.zeros(points.shape[0], dtype=float)
        total_weight = np.zeros(points.shape[0], dtype=float)
        for dx, x_weight in ((0, 1.0 - tx), (1, tx)):
            for dy, y_weight in ((0, 1.0 - ty), (1, ty)):
                ix = np.clip(ix0 + dx, 0, self.flux.shape[1] - 1)
                iy = np.clip(iy0 + dy, 0, self.flux.shape[0] - 1)
                weight = x_weight * y_weight
                valid = self.labels[iy, ix] > 0
                values += weight * valid * self.flux[iy, ix]
                total_weight += weight * valid
        values /= np.maximum(total_weight, 1.0e-30)
        return values / max(float(np.max(values)), 1.0e-30)


def solve_reference(subdiv: int = 16, tolerance: float = 1.0e-10) -> ReferenceSolution:
    labels = np.repeat(np.repeat(cell_labels(), subdiv, axis=0), subdiv, axis=1)
    h = CELL_SIZE / subdiv
    active = labels > 0
    ids = -np.ones_like(labels, dtype=np.int64)
    ids[active] = np.arange(int(active.sum()))
    diffusion, sigma_a, nu_sigma_f = material_arrays(labels)
    stiffness = lil_matrix((int(active.sum()), int(active.sum())), dtype=np.float64)
    fission = np.zeros(int(active.sum()), dtype=np.float64)

    for iy, ix in np.argwhere(active):
        row = int(ids[iy, ix])
        stiffness[row, row] += sigma_a[iy, ix] * h * h
        fission[row] = nu_sigma_f[iy, ix] * h * h
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            jx, jy = ix + dx, iy + dy
            if 0 <= jx < labels.shape[1] and 0 <= jy < labels.shape[0] and active[jy, jx]:
                d_left, d_right = diffusion[iy, ix], diffusion[jy, jx]
                face_diffusion = 2.0 * d_left * d_right / (d_left + d_right)
                stiffness[row, row] += face_diffusion
                stiffness[row, int(ids[jy, jx])] -= face_diffusion
            else:
                symmetry = ix == 0 and dx == -1 or iy == 0 and dy == -1
                if not symmetry:
                    stiffness[row, row] += 0.5 * h

    mass = csr_matrix(
        (fission, (np.arange(fission.size), np.arange(fission.size))),
        shape=(fission.size, fission.size),
    )
    eigenvalues, eigenvectors = eigsh(mass, M=stiffness.tocsr(), k=1, which="LA", tol=tolerance)
    flux = np.zeros_like(labels, dtype=np.float64)
    flux[active] = eigenvectors[:, 0]
    if float(np.mean(flux[active])) < 0.0:
        flux = -flux
    flux /= float(np.max(flux))
    return ReferenceSolution(float(eigenvalues[0]), subdiv, labels, flux)


def save_reference(solution: ReferenceSolution, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        keff=np.asarray(solution.keff),
        subdiv=np.asarray(solution.subdiv),
        geometry_sha256=np.asarray(geometry_sha256()),
        labels=solution.labels,
        flux=solution.flux,
    )


def load_reference(path: str | Path) -> ReferenceSolution:
    with np.load(Path(path)) as data:
        return ReferenceSolution(
            keff=float(data["keff"]),
            subdiv=int(data["subdiv"]),
            labels=data["labels"],
            flux=data["flux"],
        )
