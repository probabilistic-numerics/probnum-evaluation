"""Work-precision diagrams and calibration analysis for finite-dimensional problems."""

from ._calibration_measures import anees, nci
from ._error_measures import (
    mae,
    mean_error,
    relative_mae,
    relative_mean_error,
    relative_rmse,
    rmse,
)

__all__ = [
    "rmse",
    "anees",
    "nci",
    "rmse",
    "relative_rmse",
    "mae",
    "relative_mae",
    "mean_error",
    "relative_mean_error",
]
