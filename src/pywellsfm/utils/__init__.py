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
from .interpolation import (
    Interpolator,
    LinearInterpolator,
    LowerBoundInterpolator,
    PolynomialInterpolator,
    UpperBoundInterpolator,
)
from .logging_utils import (
    DEBUG,
    ERROR,
    INFO,
    WARNING,
    clear_stored_logs,
    configure_logging,
    export_stored_logs,
    export_stored_logs_to_json_file,
    export_stored_logs_to_text_file,
    get_logger,
    get_stored_log_messages,
    get_stored_logs,
    set_log_level,
)
from .plot import (
    DEFAULT_LITHO_COLORS,
    plot_accommodation,
    plot_litho_log,
    plot_water_depth,
    plot_wd_thickness_ratio,
    plot_well_analysis,
    plot_well_comparison,
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
    "LowerBoundInterpolator",
    "PolynomialInterpolator",
    "UpperBoundInterpolator",
    "wasserstein2_distance",
    "clear_stored_logs",
    "configure_logging",
    "export_stored_logs",
    "export_stored_logs_to_json_file",
    "export_stored_logs_to_text_file",
    "get_logger",
    "get_stored_log_messages",
    "get_stored_logs",
    "set_log_level",
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
    "DEFAULT_LITHO_COLORS",
    "plot_accommodation",
    "plot_litho_log",
    "plot_water_depth",
    "plot_wd_thickness_ratio",
    "plot_well_analysis",
    "plot_well_comparison",
]
