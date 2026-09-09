"""Display conventions shared by the manuscript table builders."""

from __future__ import annotations

import math
from decimal import Decimal


def tex_time_seconds(value: float) -> str:
    """Format a recorded duration with three significant digits.

    Scientific notation keeps the precision explicit for large or very small
    durations. This function affects display only; statistics use raw values.
    """
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("A duration must be finite and nonnegative.")
    if seconds == 0:
        return "0"
    mantissa, exponent_text = f"{seconds:.2e}".split("e")
    exponent = int(exponent_text)
    if -3 <= exponent <= 2:
        rounded = Decimal(mantissa).scaleb(exponent)
        return f"{rounded:.{2 - exponent}f}"
    return rf"\({mantissa}\times 10^{{{exponent}}}\)"
