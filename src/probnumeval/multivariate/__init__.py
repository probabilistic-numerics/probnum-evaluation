"""Error analysis and calibration analysis for finite-dimensional problems."""

from ._calibration_measures import anees, inclination_index, nci
from ._error_measures import (
    mae,
    max_error,
    mean_error,
    relative_mae,
    relative_max_error,
    relative_mean_error,
    relative_rmse,
    rmse,
)
from ._sample_analysis import (
    gaussianity_p_value,
    sample_reference_distance,
    sample_sample_distance,
)

__all__ = [
    "anees",
    "nci",
    "inclination_index",
    "rmse",
    "relative_rmse",
    "mae",
    "relative_mae",
    "max_error",
    "relative_max_error",
    "mean_error",
    "relative_mean_error",
    "gaussianity_p_value",
    "sample_reference_distance",
    "sample_sample_distance",
]
