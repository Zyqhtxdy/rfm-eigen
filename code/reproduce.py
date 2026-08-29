"""Reproduce the numerical results of the paper.

Each function below produces one thing the paper prints.  Run them from a
Python session started in this directory::

    >>> import reproduce
    >>> reproduce.table_2()
    >>> reproduce.figure_3()

or run every asset in one go::

    python reproduce.py

The two groups differ in what they cost.  The ``table_`` and ``figure_``
functions read the recorded output in ``data`` and write into ``../figures``
and ``data``; they take seconds.  The ``example_`` functions recompute a whole
experiment from its seeds and take between a minute and several hours, and the
three neural baselines of Example 4 take days on one card, which is why their
recorded output is committed rather than left to be recomputed.

Set the thread count before starting Python, not inside it: the threading
libraries read it when they load and ignore later changes, and a timing
comparison recorded at a thread count other than the one it reports is not a
comparison.

    RFMEIG_THREADS=6 python reproduce.py
"""

from __future__ import annotations

import sys

# ---------------------------------------------------------------------------
# the tables and figures, from the recorded output
# ---------------------------------------------------------------------------


def _assets():
    from paper_assets import assets

    assets.use_plot_style()
    return assets


def figure_1() -> None:
    """The H^1 eigenspace gap and the cluster eigenvalue error against N.

    Writes ``figures/experiment1_rewrite_overview.pdf`` and, from the same
    summary, ``data/experiment1_rewrite_summary_table.tex``.
    """
    _assets().plot_experiment1()


def table_1() -> None:
    """Empirical error statistics for the double eigenvalue of Example 1."""
    _assets().plot_experiment1()


def figure_2() -> None:
    """Effect of perturbing the six modes below the target level."""
    from paper_assets import redesign_figures
    from paper_assets.paths import FIGURES

    redesign_figures.use_publication_style()
    redesign_figures.plot_experiment1_ablation(FIGURES)


def table_2() -> None:
    """The three cost-accuracy curves of Example 2."""
    _assets().build_experiment2_table()


def figure_3() -> None:
    """Maximum relative error against computation time, Example 2."""
    _assets().plot_experiment2()


def table_3() -> None:
    """Relative errors of the first three ordered eigenvalues, Example 3."""
    _assets().build_experiment3_table()


def figure_4() -> None:
    """The first three ordered Ritz values against the feature count."""
    _assets().plot_experiment3()


def example_4_assets() -> None:
    """Table 5 and Figures 5 and 6: the IAEA quarter-core benchmark.

    Needs the recorded reference and checkpoints under ``artifacts``.
    """
    _assets().build_experiment4_assets()


def table_6() -> None:
    """The shifted Gross-Pitaevskii problem, Example 5."""
    _assets().build_experiment5_table()


def table_7() -> None:
    """The rotating two-component dipolar condensate, Example 6."""
    _assets().build_experiment6_table()


def appendix_figures() -> None:
    """Figures A.7 and A.8: the training histories of the neural baselines."""
    from paper_assets import redesign_figures
    from paper_assets.paths import FIGURES

    redesign_figures.use_publication_style()
    redesign_figures.plot_appendix_convergence(FIGURES)


def chinese_tables() -> None:
    """The translated tables, derived from the English ones beside them."""
    from paper_assets import chinese_tables as builder

    builder.build_chinese_tables()


def all_assets() -> None:
    """Every table and figure the paper includes."""
    for step in (
        figure_1,
        figure_2,
        table_2,
        figure_3,
        table_3,
        figure_4,
        example_4_assets,
        table_6,
        table_7,
        appendix_figures,
    ):
        print(f"  {step.__name__} ...", flush=True)
        step()


# ---------------------------------------------------------------------------
# the experiments, from their seeds
# ---------------------------------------------------------------------------


def example_1(run_id: str = "rerun") -> None:
    """Section 4.1: a non-extremal double eigenvalue on the square.

    100 draws at each of eight feature counts; minutes.
    """
    from rfmeig.experiments import exp1_interior_double

    exp1_interior_double.main(["--run-id", run_id])


def example_2(run_id: str = "rerun") -> None:
    """Section 4.1: a radially graded problem on the unit ball.

    Three methods on one clock, 20 draws at each of five budgets; about ten
    minutes, of which the finest mesh is three.
    """
    from rfmeig.experiments import exp2_graded_ball

    exp2_graded_ball.main(["--run-id", run_id])


def example_3(run_id: str = "rerun") -> None:
    """Section 4.1: a ten-dimensional Schrodinger eigenproblem.

    Quasi-Monte Carlo assembly on 200,000 points; hours.
    """
    from rfmeig.experiments import exp3_high_dimension

    exp3_high_dimension.main(["--run-id", run_id])


def example_4(run_id: str = "rerun") -> None:
    """Section 4.1: the IAEA quarter-core benchmark, sampled space only.

    The three neural baselines are not rerun here; their recorded histories
    and checkpoints are under ``artifacts``, and retraining them takes days.
    """
    from rfmeig.experiments import exp4_iaea

    exp4_iaea.main(["--run-id", run_id])


def example_5(run_id: str = "rerun") -> None:
    """Section 4.2: a Gross-Pitaevskii ground state."""
    from rfmeig.experiments import exp5_gpe

    exp5_gpe.main(["--run-id", run_id])


def example_6(run_id: str = "rerun") -> None:
    """Section 4.2: a rotating two-component dipolar condensate."""
    from rfmeig.experiments import exp6_dipolar

    exp6_dipolar.main(["--run-id", run_id])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for name in sys.argv[1:]:
            globals()[name]()
    else:
        all_assets()
