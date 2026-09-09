# Reproducibility repository

This repository contains the code and the recorded output for the article

> **High-probability convergence of the random feature method for
> multiple eigenvalues of elliptic operators**

The article's five numerical result tables and six figures are generated from
the recorded output included here. The material-parameter table is written
directly in the manuscript.

## Reproducing

To reproduce the numerical experiments you need Python 3.12 and the packages
pinned in [`code/requirements.txt`](code/requirements.txt). See
[`code/README.md`](code/README.md) for instructions to reproduce the results,
and for information about postprocessing.

The figures the article includes are in [`figures`](figures); the tables it
inputs are the `.tex` files in [`code/data`](code/data). Both are regenerated
by the scripts in [`code`](code).

## Citing

If you use these results, please cite both the article and this repository.

## Disclaimer

Everything is provided as is and without warranty.
