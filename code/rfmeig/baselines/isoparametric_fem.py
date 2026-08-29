r"""Isoparametric :math:`P_2` finite elements, the mesh baseline of Example 2.

On a curved domain a straight-sided mesh caps the attainable accuracy no matter
how fine it is, so the boundary nodes of the quadratic mesh are projected onto
the sphere.  That is the fair form of the baseline: without it the comparison
would be against a method carrying an avoidable geometric error, and the
manuscript's claim that the sampled space is more accurate would be measuring
the mesh's geometry rather than its approximation.

The bilinear form is the graded one,
:math:`\int_\Omega a\nabla u\cdot\nabla v`, so the baseline solves the same
problem as the sampled space rather than a constant-coefficient neighbour.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg
from skfem import Basis, BilinearForm, ElementTetP2, MeshTet, MeshTet2
from skfem.helpers import dot, grad
from skfem.models.poisson import laplace, mass


@BilinearForm
def graded_laplace(u, v, w):
    """:math:`\\int a\\nabla u\\cdot\\nabla v` with ``a`` at the quadrature points."""
    return w["conductivity"] * dot(grad(u), grad(v))


@dataclass
class Result:
    values: np.ndarray
    dimension: int
    seconds: float
    assemble_seconds: float
    solve_seconds: float


def curved_ball(subdivisions: int, radius: float = 1.0) -> MeshTet2:
    """The built-in tetrahedral ball, refined, made quadratic, and curved.

    The extra nodes of a quadratic mesh are edge midpoints, indexed after the
    vertices.  A midpoint belongs to the curved boundary exactly when both
    endpoints of its edge do, and moving those -- and only those -- is what
    turns a chord into an arc.  Without this step the boundary stays polyhedral
    and caps the eigenvalue error whatever the element.
    """
    linear = MeshTet.init_ball(subdivisions)
    quadratic = MeshTet2.from_mesh(linear)
    points = quadratic.doflocs.copy()
    vertices = linear.p.shape[1]
    distance = np.linalg.norm(points, axis=0)
    on_sphere = np.abs(distance[:vertices] - radius) < 1.0e-10

    edges = linear.edges
    both_ends = on_sphere[edges[0]] & on_sphere[edges[1]]
    midpoints = vertices + np.flatnonzero(both_ends)
    if midpoints.size:
        block = points[:, midpoints]
        points[:, midpoints] = radius * block / np.linalg.norm(block, axis=0)
    return MeshTet2(points, quadratic.t)


def solve(domain, subdivisions: int, count: int) -> Result:
    """Assemble and solve the Dirichlet eigenproblem on one curved mesh.

    The lowest ``count`` eigenvalues are taken by shift-inverted Lanczos at
    shift zero, which is the standard way of reaching the bottom of a
    generalized pencil without forming the whole spectrum.
    """
    started = time.perf_counter()
    mesh = curved_ball(subdivisions, domain.radius)
    basis = Basis(mesh, ElementTetP2())

    # the coefficient read at the quadrature points the form is assembled on
    coordinates = basis.global_coordinates().value
    conductivity = domain.coefficient(
        coordinates.reshape(domain.dimension, -1).T
    ).reshape(coordinates.shape[1:])
    if np.allclose(conductivity, 1.0):
        stiffness = laplace.assemble(basis)
    else:
        stiffness = graded_laplace.assemble(basis, conductivity=conductivity)
    mass_matrix = mass.assemble(basis)

    interior = basis.complement_dofs(basis.get_dofs())
    stiffness_c = stiffness[interior][:, interior]
    mass_c = mass_matrix[interior][:, interior]
    assembled = time.perf_counter()

    values = scipy.sparse.linalg.eigsh(
        stiffness_c,
        k=count,
        M=mass_c,
        sigma=0.0,
        which="LM",
        return_eigenvectors=False,
    )
    values = np.sort(np.asarray(values))
    finished = time.perf_counter()
    return Result(
        values=values,
        dimension=int(stiffness_c.shape[0]),
        seconds=finished - started,
        assemble_seconds=assembled - started,
        solve_seconds=finished - assembled,
    )
