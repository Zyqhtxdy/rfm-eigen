r"""The condensate benchmarks, as their source papers state them.

Each entry records the domain, the interaction strength, the potential, and --
where the source prints them -- the eigenvalue and the energy it reports,
together with the method and the resolution those came from.  Keeping the
source's own numbers beside the problem is what lets a run be checked against a
printed table rather than against a value this code produced.

Example 5 uses ``E7_shifted_400``: a shifted Gaussian potential with
:math:`\beta=400`, whose reported values come from a weak Galerkin method with
linear elements on a mesh of a hundred and twenty-eight subdivisions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class PaperBenchmark:
    name: str
    paper_id: str
    dimension: int
    bounds: tuple[tuple[float, float], ...]
    beta: float
    potential_kind: str
    reference_lambda: float | None = None
    reference_energy: float | None = None
    source_method: str = ""
    source_metric: str = ""
    disorder_seed: int | None = None

    def potential(self, points: Array) -> Array:
        squared_radius = np.sum(points * points, axis=1)
        if self.potential_kind == "harmonic_half":
            return 0.5 * squared_radius
        if self.potential_kind == "harmonic":
            return squared_radius
        if self.potential_kind == "constant_one":
            return np.ones(points.shape[0])
        if self.potential_kind == "periodic_product_3d":
            return np.prod(
                np.sin(0.25 * np.pi * points) ** 2, axis=1
            )
        if self.potential_kind == "shifted_gaussian":
            x = points[:, 0]
            y = points[:, 1]
            return squared_radius + 8.0 * np.exp(-(x - 1.0) ** 2 - y * y)
        if self.potential_kind == "lattice_hho":
            x = points[:, 0]
            y = points[:, 1]
            return (
                0.5 * squared_radius
                + 15.0
                * (
                    1.0
                    + np.sin(0.5 * np.pi * x)
                    * np.sin(0.5 * np.pi * y)
                )
            )
        if self.potential_kind == "optical_3d":
            optical = np.sum(
                np.sin(0.25 * np.pi * points) ** 2, axis=1
            )
            return squared_radius + 100.0 * optical
        if self.potential_kind == "disorder_10_50":
            return self._disorder(points, low=10.0, high=50.0, cells=16)
        if self.potential_kind == "disorder_0_1024":
            return self._disorder(points, low=0.0, high=1024.0, cells=32)
        raise ValueError(self.potential_kind)

    def _disorder(
        self, points: Array, *, low: float, high: float, cells: int
    ) -> Array:
        if self.dimension != 2 or self.disorder_seed is None:
            raise ValueError("Disorder benchmarks require a 2D seed")
        rng = np.random.default_rng(self.disorder_seed)
        values = rng.choice([low, high], size=(cells, cells))
        indices = []
        for axis, (lower, upper) in enumerate(self.bounds):
            coordinate = (points[:, axis] - lower) / (upper - lower)
            indices.append(
                np.clip((cells * coordinate).astype(int), 0, cells - 1)
            )
        return values[indices[0], indices[1]]


PAPER_BENCHMARKS = {
    "E1_periodic_10": PaperBenchmark(
        name="E1_periodic_10",
        paper_id="E1",
        dimension=3,
        bounds=((-16.0, 16.0),) * 3,
        beta=10.0,
        potential_kind="periodic_product_3d",
        reference_lambda=0.143834048046,
        reference_energy=0.071660785256,
        source_method="Q10 spectral elements on a 100^3 mesh",
        source_metric="lambda and energy",
    ),
    "E1_periodic_4000": PaperBenchmark(
        name="E1_periodic_4000",
        paper_id="E1",
        dimension=3,
        bounds=((-16.0, 16.0),) * 3,
        beta=4000.0,
        potential_kind="periodic_product_3d",
        reference_lambda=0.34919956116,
        reference_energy=0.127936543199,
        source_method="Q10 spectral elements on a 100^3 mesh",
        source_metric="lambda and energy",
    ),
    "E1_optical_3d": PaperBenchmark(
        name="E1_optical_3d",
        paper_id="E1",
        dimension=3,
        bounds=((-8.0, 8.0),) * 3,
        beta=1600.0,
        potential_kind="optical_3d",
        reference_lambda=80.89511440602,
        reference_energy=33.80227900547,
        source_method="Q40 spectral elements and Sobolev gradient flow",
        source_metric="lambda, energy, nonlinear residual, iterations",
    ),
    "E2_harmonic_1000": PaperBenchmark(
        name="E2_harmonic_1000",
        paper_id="E2",
        dimension=2,
        bounds=((-8.0, 8.0),) * 2,
        beta=1000.0,
        potential_kind="harmonic_half",
        source_method="RT0 mixed FEM and J-method",
        source_metric="state/flux and energy/eigenvalue convergence",
    ),
    "E2_harmonic_1": PaperBenchmark(
        name="E2_harmonic_1",
        paper_id="E2",
        dimension=2,
        bounds=((-8.0, 8.0),) * 2,
        beta=1.0,
        potential_kind="harmonic_half",
        source_method="RT0 mixed FEM and J-method",
        source_metric="state/flux and energy/eigenvalue convergence",
    ),
    "E2_harmonic_10": PaperBenchmark(
        name="E2_harmonic_10",
        paper_id="E2",
        dimension=2,
        bounds=((-8.0, 8.0),) * 2,
        beta=10.0,
        potential_kind="harmonic_half",
        source_method="RT0 mixed FEM and J-method",
        source_metric="state/flux and energy/eigenvalue convergence",
    ),
    "E2_harmonic_100": PaperBenchmark(
        name="E2_harmonic_100",
        paper_id="E2",
        dimension=2,
        bounds=((-8.0, 8.0),) * 2,
        beta=100.0,
        potential_kind="harmonic_half",
        source_method="RT0 mixed FEM and J-method",
        source_metric="state/flux and energy/eigenvalue convergence",
    ),
    "E3_lattice_1000": PaperBenchmark(
        name="E3_lattice_1000",
        paper_id="E3",
        dimension=2,
        bounds=((-8.0, 8.0),) * 2,
        beta=1000.0,
        potential_kind="lattice_hho",
        source_method="modified hybrid high-order method",
        source_metric="lower-energy and eigenvalue convergence",
    ),
    "E3_disorder": PaperBenchmark(
        name="E3_disorder",
        paper_id="E3",
        dimension=2,
        bounds=((-8.0, 8.0),) * 2,
        beta=1.0,
        potential_kind="disorder_10_50",
        source_method="modified hybrid high-order method",
        source_metric="Anderson localization and lower energy",
        disorder_seed=250619944,
    ),
    "E6_disorder": PaperBenchmark(
        name="E6_disorder",
        paper_id="E6",
        dimension=2,
        bounds=((-1.0, 1.0),) * 2,
        beta=1.0,
        potential_kind="disorder_0_1024",
        source_method="positivity-preserving lumped P1 FEM",
        source_metric="localization and convergence orders",
        disorder_seed=240517090,
    ),
    "E7_shifted_400": PaperBenchmark(
        name="E7_shifted_400",
        paper_id="E7",
        dimension=2,
        bounds=((-8.0, 8.0),) * 2,
        beta=400.0,
        potential_kind="shifted_gaussian",
        reference_lambda=16.629569,
        reference_energy=5.850096,
        source_method="weak Galerkin method, k=1, N=128",
        source_metric="lambda and energy",
    ),
    "E7_flat_1": PaperBenchmark(
        name="E7_flat_1",
        paper_id="E7",
        dimension=2,
        bounds=((-8.0, 8.0),) * 2,
        beta=1.0,
        potential_kind="constant_one",
        reference_lambda=1.085733,
        reference_energy=0.540723,
        source_method="quadratic weak Galerkin method, N=64",
        source_metric="lambda and energy",
    ),
    "E7_harmonic_2": PaperBenchmark(
        name="E7_harmonic_2",
        paper_id="E7",
        dimension=2,
        bounds=((-1.0, 1.0),) * 2,
        beta=2.0,
        potential_kind="harmonic",
        reference_lambda=6.300447,
        reference_energy=2.876855,
        source_method="quadratic weak Galerkin method, N=64",
        source_metric="lambda and energy",
    ),
}


def get_paper_benchmark(name: str) -> PaperBenchmark:
    try:
        return PAPER_BENCHMARKS[name]
    except KeyError as error:
        raise KeyError(f"Unknown paper benchmark: {name}") from error
