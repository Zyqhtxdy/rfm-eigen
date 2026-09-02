# Instructions to reproduce

## Setup

```bash
python -m pip install -r requirements.txt
```

The versions in `requirements.txt` are the ones the recorded output was
produced under. The article quotes wall-clock times, so the library that does
the dense eigensolve is part of the measurement: a different LAPACK moves the
seconds, and at the `1e-12` truncation threshold it can move the retained rank
by a few directions.

Set the thread count in the environment before starting Python. The threading
libraries read it when they load and ignore later changes, and the experiment
drivers refuse to run at a count other than the one they were asked for.

```bash
export RFMEIG_THREADS=6      # Windows: $env:RFMEIG_THREADS = 6
```

## Rebuilding the tables and figures

These read the recorded output in `data` and write into `../figures` and
`data`. They take seconds.

```bash
python reproduce.py
```

or, one at a time, from a session started in this directory:

```python
>>> import reproduce
>>> reproduce.figure_1()            # H^1 gap and eigenvalue error against N
>>> reproduce.table_1()             # error statistics, Example 1
>>> reproduce.figure_2()            # the lower-mode perturbation
>>> reproduce.table_2()             # the three cost-accuracy curves
>>> reproduce.figure_3()            # error against time, Example 2
>>> reproduce.table_3()             # ten-dimensional errors
>>> reproduce.figure_4()            # the first three ordered Ritz values
>>> reproduce.example_4_assets()    # Table 5 and Figures 5 and 6
>>> reproduce.table_6()             # the Gross-Pitaevskii ground state
>>> reproduce.table_7()             # the dipolar condensate
>>> reproduce.appendix_figures()    # Figures A.7 and A.8
```

Rebuilding is byte-identical: the six `data/*_table.tex` files are unchanged by
a rebuild against the committed output.

## Checking that the tables still follow from the runs

```bash
python audit.py
```

This rebuilds every table whose builder runs from the committed data and
compares it byte for byte with the copy in `data`, and it compares the number
of draws each caption states with the number the corresponding run holds.  The
second check is the one worth having: a caption that says twenty over a run of
five is invisible to any hash.  The result is written to
`data/reproducibility_audit.json`, and the command exits non-zero on failure.

`data/core_artifact_sha256.csv` records the hash of every committed input, so a
file that changed after the fact can be found without rerunning anything.

## Rerunning the experiments

These recompute an experiment from its seeds and write a sealed run directory
under `../output`, beside `code`, carrying the code hash, the configuration, the library versions
and the thread count. They do not overwrite `data`.

```python
>>> reproduce.example_1()   # 100 draws at eight feature counts; minutes
>>> reproduce.example_2()   # three methods on one clock; about ten minutes
>>> reproduce.example_3()   # quasi-Monte Carlo assembly; hours
>>> reproduce.example_4()   # the sampled space only, see below
>>> reproduce.example_5()
>>> reproduce.example_6()
```

Equivalently, from the command line:

```bash
python -m rfmeig.experiments.exp2_graded_ball --run-id my-run
```

**The three neural baselines of Example 4 are not rerun by any of these.**
Training them took between 4.2e3 and 5.4e4 seconds each on one card, so their
histories and selected checkpoints are committed under `artifacts` and the
asset builder reads them. `reproduce.example_4()` recomputes the sampled space
against the same recorded reference.

Example 4's pipeline has entry points of its own, for the steps `reproduce.py`
does not cover: `python -m iaea_repro.cli` generates the finite-volume
reference, trains the three neural baselines, runs the feature search, and
checks each baseline against the published errors before the comparison is
made.

Timings are only comparable when nothing else is running on the machine. The
times in Tables 2, 5, 6 and 7 are medians over repeated solves of the same
configuration; they cover assembly and solution, and the error evaluation lies
outside the timer.

## Layout

```text
rfmeig/            the method, the problems, and the baselines
  experiments/     one module per numerical example, each with a __main__
  baselines/       every method the article compares against
  problems/        the domains, coefficients and exact spectra
  rayleigh_ritz.py the whitened Ritz solves the experiments share
  provenance.py    run directories, code hashes, thread pinning
iaea_repro/        Example 4: the reactor solvers and their assets
paper_assets/      the builders that write the tables and the figures
data/              the recorded output, and the generated .tex tables
artifacts/         Example 4 references, checkpoints and histories
configs/           Example 4 configurations
```

The baselines are in `rfmeig/baselines`: isoparametric `P2` finite elements and
Eig-PIELM for Example 2, Deep Ritz for Example 3, the three neural reactor
methods for Example 4, weak Galerkin for Example 5, and GFLM-KTM for Example 6.
