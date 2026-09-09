"""Reproduce the numerical results of the paper.

Each function below produces one thing the paper prints.  Run them from a
Python session started in this directory::

    >>> import reproduce
    >>> reproduce.table_2()
    >>> reproduce.figure_2()

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

import argparse
import importlib

# ---------------------------------------------------------------------------
# the tables and figures, from the recorded output
# ---------------------------------------------------------------------------


def _assets():
    from paper_assets import assets
    from paper_assets.paths import DATA, FIGURES

    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
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
    _assets().build_experiment1_table()


def table_2() -> None:
    """The three cost-accuracy curves of Example 2."""
    _assets().build_experiment2_table()



def figure_2() -> None:
    """The first three eigenvalue estimates against the feature count."""
    _assets().plot_experiment3()


def example_4_assets() -> None:
    """Table 4 and Figures 3 and 4: the IAEA quarter-core benchmark.

    Needs the recorded reference and checkpoints under ``artifacts``.
    """
    _assets().build_experiment4_assets()


def example_4_figures() -> None:
    """Redraw Figures 3 and 4 from recorded states, preserving all tables."""
    from iaea_repro.paper_assets import generate_experiment4_assets
    from paper_assets.paths import DATA, FIGURES, ROOT

    generate_experiment4_assets(ROOT, FIGURES, figures_only=True, data_directory=DATA)


def table_5() -> None:
    """The shifted Gross-Pitaevskii problem, Example 5."""
    _assets().build_experiment5_table()


def table_6() -> None:
    """The rotating two-component dipolar condensate, Example 6."""
    _assets().build_experiment6_table()


def appendix_figures() -> None:
    """Appendix figures: the training histories of the neural baselines."""
    from paper_assets import redesign_figures
    from paper_assets.paths import FIGURES

    redesign_figures.use_publication_style()
    redesign_figures.plot_appendix_convergence(FIGURES)


def chinese_tables() -> None:
    """The translated tables, derived from the English ones beside them."""
    from paper_assets import chinese_tables as builder

    builder.build_chinese_tables()


def all_assets() -> None:
    """The five numerical result tables and six figures generated from the data."""
    for step in (
        figure_1,
        table_2,
        figure_2,
        example_4_assets,
        table_5,
        table_6,
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
    _run_example("exp1_interior_double", run_id)


def example_2(run_id: str = "rerun") -> None:
    """Example 2: a radially graded problem on the unit ball.

    Three methods on one clock, 20 draws at each of five budgets; about ten
    minutes, of which the finest mesh is three.
    """
    _run_example("exp2_graded_ball", run_id)


def example_3(run_id: str = "rerun") -> None:
    """Example 3: a ten-dimensional Schrodinger eigenproblem.

    Quasi-Monte Carlo assembly on 2**22 points and independent final integration; hours.
    """
    _run_example("exp3_high_dimension", run_id)


def example_4(run_id: str = "rerun") -> None:
    """Example 4: the IAEA quarter-core benchmark, sampled space only.

    The three neural baselines are not rerun here; their recorded histories
    and checkpoints are under ``artifacts``, and retraining them takes days.
    """
    _run_example("exp4_iaea", run_id)


def example_5(run_id: str = "rerun") -> None:
    """Example 5: a Gross-Pitaevskii ground state."""
    _run_example("exp5_gpe", run_id)


def example_6(run_id: str = "rerun") -> None:
    """Example 6: a rotating two-component dipolar condensate."""
    _run_example("exp6_dipolar", run_id)


def _run_example(module_name: str, run_id: str) -> None:
    from rfmeig import PINNED_THREADS

    module = importlib.import_module(f"rfmeig.experiments.{module_name}")
    module.main(["--run-id", run_id, "--threads", str(PINNED_THREADS)])


def main(argv: list[str] | None = None) -> None:
    commands = {
        function.__name__: function for function in (
            figure_1, table_1, table_2, figure_2, example_4_assets, example_4_figures,
            table_5, table_6, appendix_figures, chinese_tables, all_assets,
            example_1, example_2, example_3, example_4, example_5, example_6,
        )
    }
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("commands", nargs="*", metavar="COMMAND", help=", ".join(commands))
    args = parser.parse_args(argv)
    for name in args.commands:
        if name not in commands:
            parser.error(f"unknown command {name!r}; choose from {', '.join(commands)}")
    for name in args.commands or ["all_assets"]:
        commands[name]()


if __name__ == "__main__":
    main()
