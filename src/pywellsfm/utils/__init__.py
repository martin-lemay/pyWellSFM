"""I/O public API.

This package contains file-format specific loaders and utilities.

The symbols re-exported here form the supported, stable entry points. Callers
should prefer importing from `pywellsfm.io` instead of submodules.
"""

from .geometry import (
    IntervalDistanceMethod,
    center_distance,
    gap_distance,
    gap_overlapping_width_distance,
    gap_times_center_distance,
    hausdorff_distance,
    wasserstein2_distance,
)
from .helpers import (
    Interpolator,
    LinearInterpolator,
    PolynomialInterpolator,
)

__all__ = [
    "center_distance",
    "gap_distance",
    "gap_overlapping_width_distance",
    "gap_times_center_distance",
    "hausdorff_distance",
    "Interpolator",
    "IntervalDistanceMethod",
    "LinearInterpolator",
    "PolynomialInterpolator",
    "wasserstein2_distance",
]
