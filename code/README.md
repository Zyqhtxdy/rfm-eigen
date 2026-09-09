# Reproducing the numerical results

## Setup

Run the following commands from this `code` directory in a Python environment:

```bash
python -m pip install -r requirements.txt
```

The recorded environment uses Python 3.12 and the library versions pinned in
`requirements.txt`. For development, also install the editable package and tools:

```bash
python -m pip install -e ".[dev]"
```

Set the thread budget before starting Python. In PowerShell:

```powershell
$env:RFMEIG_THREADS = "6"
```

In a POSIX shell:

```bash
export RFMEIG_THREADS=6
```

The `rfmeig` package sets numerical-library thread variables from this value
at import. Start a fresh process before changing the budget: drivers check both
the environment and loaded BLAS/OpenMP pools. Direct experiment commands must
also receive the matching `--threads 6`; `reproduce.py` passes it automatically.
Recorded wall times depend on the hardware, thread budget and numerical-library
build. Small changes near the `1e-12` truncation threshold can also change the
retained rank.

## Rebuild the manuscript assets

```bash
python reproduce.py
```

This reads `data` and `artifacts/experiment4`, then generates five numerical
result tables in `data` and six PDFs in `../figures`. The material-parameter
table is written directly in the manuscript. To rebuild selected assets:

| Command after `python reproduce.py` | Output |
| --- | --- |
| `figure_1` | Example 1 mean-error curves and its result table |
| `table_1` | Example 1 result table |
| `table_2` | Example 2 comparison table |
| `figure_2` | Example 3 independently evaluated eigenvalue estimates |
| `example_4_assets` | Example 4 comparison table and two figures |
| `example_4_figures` | Example 4 material map and flux-error comparison |
| `table_5` | Example 5 result table |
| `table_6` | Example 6 result table |
| `appendix_figures` | Two neural-training history figures |
| `chinese_tables` | Translated captions and headers under `data/zh` |

Multiple command names can be supplied in one invocation. `python reproduce.py
--help` lists all public commands. The two Example 4 figures use a compact
material map and four square error panels with a shared colorbar.

For an isolated rebuild, set an output directory before starting Python:

```powershell
$env:RFM_REPRO_OUTPUT_ROOT = "D:/NonLinear/artifacts/rfmeig-assets-check"
python reproduce.py all_assets chinese_tables
```

Outputs go under that directory's `data` and `figures`; recorded inputs
are still read from this checkout. In a POSIX shell, use `export
RFM_REPRO_OUTPUT_ROOT=/path/to/assets-check` instead. Clear the variable to
restore the normal output locations.

## Check the recorded results

```bash
python audit.py
```

The audit independently recomputes Example 1's mean, median and 90th-percentile
summaries and its calibration constants; checks Example 3's eigenvalue summaries;
verifies draw counts, available archive hashes and core input hashes; and
rebuilds all five English result tables in a temporary directory for bytewise
comparison. The tables in `data` are not overwritten. The JSON report goes to
the active output data directory, or to `--output PATH`, and failures produce a
nonzero exit status. `data/core_artifact_sha256.csv` lists the selected core
inputs covered by the hash check. The partial run exports are described in
`data/runs/README.md`.
The repository's `.gitattributes` preserves the bytes of recorded data,
artifacts and configurations across platforms, including their line endings.

Example 1 uses composite Gauss rules of order 512 per subinterval and coordinate
for assembly, field evaluation and subspace comparison. Its validation set has
100 draws at each of eight feature counts; the independent calibration set has
20 draws at each count. The plotted statistic is the arithmetic mean. The
full-precision reference value is `149.91769837259983`. Median and 90th-percentile
statistics remain in the recorded summaries. The quadrature-check files retain
all 2,880 evaluations at orders 128, 256 and 512, with their configuration and
source provenance in `experiment1_refinement_provenance.json`.

Development checks:

```bash
python -m ruff check .
python -m pytest -q
```

The tests cover matrix assembly, whitening, separable quadrature, recorded
Example 1 draws, independent reference backends, run recovery and sealing, and
table reconstruction. They use small problems and recorded data.

## Rerun experiments

```bash
python reproduce.py example_1
```

Replace `example_1` with any of `example_2` through `example_6`. These wrappers
use run ID `rerun`; an existing run is protected against replacement. For a
chosen run ID and experiment-specific settings, use the corresponding module:

```bash
python -m rfmeig.experiments.exp1_interior_double --threads 6 --run-id validation-512
python -m rfmeig.experiments.exp2_graded_ball --threads 6 --run-id ball-comparison
```

New runs go to `../output/EXPERIMENT/RUN_ID`. Set `RFMEIG_RUN_ROOT` before
starting Python to choose another run root; this variable is separate from
`RFM_REPRO_OUTPUT_ROOT`, which redirects manuscript-asset output. Runs record
their configuration, source hash, library versions and thread settings.
Successful runs are sealed with an output hash manifest. A sealed run cannot
be resumed or written through the provenance API.

Use `--resume` with the same run ID to continue an interrupted, unsealed run.
The source hash and configuration must match. Completed solver calls are
reused; an interrupted call is rerun. Example 2 checkpoints each comparison
configuration. The alternate Example 4 neural driver reuses a completed
training result; it does not resume an unfinished optimizer trajectory.

The drivers' `--help` pages list feature budgets, quadrature settings, seeds and
other parameters. Runs may take minutes to hours; the three Example 4 neural
baselines have separate training commands and recorded checkpoints.
`example_4` recomputes the sampled space against the recorded finite-volume
reference. It does not retrain those neural baselines.

The full reactor pipeline is available through `python -m iaea_repro.cli
--help`. Its `reference` command writes a fresh file under the run root by
default and accepts `--output PATH`; an existing reference file is never
replaced. Recorded histories, references and selected checkpoints are under
`artifacts/experiment4`. Example 5 also retains the independent sine-spectral
reference solver in `rfmeig/baselines/sine_spectral_gpe.py`.

## Layout

```text
rfmeig/             method, numerical operators, problems and baselines
  experiments/      one command-line module per example
  provenance.py     run records, recovery, sealing and thread checks
  separable_assembly.py  tensor-product assembly for Example 1
iaea_repro/         reactor reference, baseline training and evaluation
paper_assets/       table and figure builders
tests/             numerical and reproducibility regression tests
data/              recorded numerical output and generated result tables
artifacts/         reactor references, checkpoints and training histories
configs/           reactor configurations
```


## Final evaluation records

Example 3 uses 2**22 Sobol points for RFM assembly. `cube_product_integrals`
evaluates the fixed RFM fields independently. The active 160-draw records,
fixed coefficients and configuration are in `data/runs/experiment3_independent`.
The historical run under `data/runs/experiment3` is retained as an archive.
The figure reads `data/experiment3_drm_independent.csv` for the six fixed
baseline networks. To recompute their independent evaluation without training:

```bash
python -m rfmeig.experiments.exp3_drm_final --run-id final-check --device cuda
```

This records two fixed scrambles with nested 2**22, 2**23 and 2**24 interior
rules, and 2**14, 2**15 and 2**16 points per boundary face. The reported
penalized quotient is formed from the averaged mass, energy and boundary
estimates. The final evaluation points do not select checkpoints.

Example 5 keeps the 56-point optimization and 84-point residual evaluation;
final energy and chemical potential use a mass-normalized 112-point rule.
`data/evaluation_updates` preserves the original optimizer records, fixed
states, per-draw checks and final-evaluation metadata. The 95 accepted draws
and the recorded solver times are unchanged. WG labels use subdivisions `n`;
on the 16-by-16 box, Cartesian spacing is `16/n`.

Example 6 evaluates fixed RFM fields on the reference's physical periodic
nodes for wavefunction errors. The midpoint rule for energies is unchanged.
The update records preserve all 40 field checks and the original timed rows.
