"""Independently evaluate the six fixed Ji networks, without retraining.

Two fixed scrambles and nested refinements are recorded. The reported quantity
retains the stored reproduction's boundary penalty and equal-face convention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

from rfmeig import quadrature
from rfmeig.baselines.deep_ritz_cube import load_checkpoint, potential
from rfmeig.problems.high_dimensional_cube import HighDimensionalCube


def main(argv=None):
    code = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoints", type=Path, default=code / "data/ji2024_drm_validated_checkpoints"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-id", default="final")
    parser.add_argument(
        "--potentials", nargs="+", choices=["square", "exp"], default=["square", "exp"]
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    DEVICE = args.device
    Q = 2**24
    CHECKS = [2**22, 2**23, 2**24]
    OUT = args.output or code / "output/experiment3" / args.run_id / "drm_final_integrals.jsonl"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {
            (r["potential"], r["scramble"]) for r in map(json.loads, OUT.read_text().splitlines())
        }
        for record in map(json.loads, OUT.read_text().splitlines()):
            for filename, expected in record["checkpoint_sha256"].items():
                if (
                    hashlib.sha256((args.checkpoints / filename).read_bytes()).hexdigest()
                    != expected
                ):
                    raise RuntimeError("checkpoint changed since the recorded evaluation")
    for name in args.potentials:
        problem = HighDimensionalCube(name)
        paths = [args.checkpoints / f"{name}_mode{k}.pt" for k in [1, 2, 3]]
        loaded = [load_checkpoint(p) for p in paths]
        models = [m.to(DEVICE) for m, _ in loaded]
        for seed in [91094001, 91094002]:
            if (name, seed) in done:
                continue
            started = time.time()
            points, weights = quadrature.sobol_cube(Q, 10, seed, density="beta22")
            masses = np.zeros(3)
            energies = np.zeros(3)
            gram = np.zeros((3, 3))
            answers = {}
            for start in range(0, Q, 4096):
                stop = min(start + 4096, Q)
                x = torch.as_tensor(points[start:stop], device=DEVICE).requires_grad_(True)
                w = torch.as_tensor(weights[start:stop], device=DEVICE)
                reaction = 20 * potential(name, x).sum(1)
                values = []
                for k, model in enumerate(models):
                    u = model(x)
                    gradient = torch.autograd.grad(u.sum(), x)[0]
                    masses[k] += float(torch.sum(w * u.detach().square()))
                    energies[k] += float(
                        torch.sum(
                            w * (gradient.square().sum(1) + reaction.detach() * u.detach().square())
                        )
                    )
                    values.append(u.detach())
                stacked = torch.stack(values, dim=1)
                gram += (stacked.T @ (w[:, None] * stacked)).cpu().numpy()
                if stop in CHECKS:
                    scale = Q / stop
                    answers[stop] = dict(
                        mass=(scale * masses).tolist(),
                        domain_energy=(scale * energies).tolist(),
                        gram=(scale * gram).tolist(),
                    )
                    print(
                        name, seed, "interior", stop, "seconds", time.time() - started, flush=True
                    )
            del points, weights
            # Equal face averages, preserving the stored Ji reproduction's convention.
            per_face = 2**16
            boundary_totals = {n: np.zeros(3) for n in [2**14, 2**15, 2**16]}
            for face in range(20):
                base = qmc.Sobol(9, scramble=True, seed=seed + 1000 + face).random_base2(16)
                x = np.insert(base, face // 2, float(face % 2), axis=1)
                with torch.no_grad():
                    for k, model in enumerate(models):
                        chunks = [
                            model(torch.as_tensor(x[s : s + 4096], device=DEVICE))
                            .square()
                            .cpu()
                            .numpy()
                            for s in range(0, per_face, 4096)
                        ]
                        squared = np.concatenate(chunks)
                        for n in boundary_totals:
                            boundary_totals[n][k] += squared[:n].mean() / 20
            ref = np.asarray(problem.reference_levels())
            for n, a in answers.items():
                b = boundary_totals[n // 256]
                mass = np.asarray(a["mass"])
                energy = np.asarray(a["domain_energy"])
                g = np.asarray(a["gram"])
                lam = (energy + 2000 * b) / mass
                a.update(
                    boundary_mean=b.tolist(),
                    boundary_points=20 * (n // 256),
                    lambda_values=lam.tolist(),
                    relative_errors=(abs(lam - ref) / ref).tolist(),
                    normalized_gram=(g / np.sqrt(mass[:, None] * mass[None, :])).tolist(),
                )
            row = dict(
                potential=name,
                scramble=seed,
                density="beta22",
                reference=ref.tolist(),
                evaluation=answers,
                metadata=[d for _, d in loaded],
                checkpoint_sha256={
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
                },
                seconds=time.time() - started,
            )
            with OUT.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(name, seed, "DONE", answers[Q]["lambda_values"], flush=True)
        del models
        torch.cuda.empty_cache()
    print("DONE six fixed networks, two independent scrambles", flush=True)


if __name__ == "__main__":
    main()
