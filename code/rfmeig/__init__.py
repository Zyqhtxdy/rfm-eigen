"""Numerical experiments for the random feature method applied to isolated
multiple elliptic eigenvalues.

The shared numerical layer is :mod:`rfmeig.quadrature`, :mod:`rfmeig.features`,
:mod:`rfmeig.rayleigh_ritz` and :mod:`rfmeig.metrics`; the problems and the
methods compared against are under :mod:`rfmeig.problems` and
:mod:`rfmeig.baselines`; and one module of :mod:`rfmeig.experiments` reproduces
each of the numerical examples of Section 4.

The manuscript calls them Examples and this package calls them experiments, so
the map is written out once here:

=============  ==============================================================
Example 1      :mod:`rfmeig.experiments.exp1_interior_double`
Example 2      :mod:`rfmeig.experiments.exp2_graded_ball`
Example 3      :mod:`rfmeig.experiments.exp3_high_dimension`
Example 4      :mod:`rfmeig.experiments.exp4_iaea`
Example 5      :mod:`rfmeig.experiments.exp5_gpe`
Example 6      :mod:`rfmeig.experiments.exp6_dipolar`
=============  ==============================================================

Threads are pinned here, before anything below imports NumPy.  The threading
libraries read these variables when they are loaded and ignore later changes, so
a thread count set after the first ``import numpy`` has no effect at all -- and a
timing comparison run at a thread count other than the one it reports is not a
comparison.  ``RFMEIG_THREADS`` sets it; the experiment drivers verify that the
count actually in force is the one they were asked for, and stop if it is not.
"""

import os

#: The thread count every experiment runs at unless told otherwise.  One thread
#: is the default because it is the setting the recorded comparisons were made
#: at: a baseline whose cost is not dense linear algebra gains nothing from more,
#: so a multi-threaded run flatters whichever side is dense.
DEFAULT_THREADS = 1

#: The variables the threading libraries read at load time.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _pin_threads() -> int:
    """Fix the thread count before NumPy is imported anywhere below.

    An explicit ``OMP_NUM_THREADS`` in the environment wins, so that a launcher
    or a scheduler can still decide; otherwise ``RFMEIG_THREADS`` decides, and
    failing that the default above.
    """
    requested = os.environ.get("RFMEIG_THREADS") or os.environ.get("OMP_NUM_THREADS")
    count = int(requested) if requested else DEFAULT_THREADS
    for name in THREAD_VARIABLES:
        os.environ[name] = str(count)
    return count


#: The count actually in force, recorded at import so that a run can state it.
PINNED_THREADS = _pin_threads()

__version__ = "1.0.0"
