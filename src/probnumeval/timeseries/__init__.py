"""Work-precision diagrams and calibration analysis for time series problems.

Applicable to (O)DE solvers, filters, smoothers, and more.

Some sources:
http://folk.ntnu.no/skoge/prost/proceedings/ifac2002/data/content/02779/2779.pdf
https://iopscience.iop.org/article/10.1088/1742-6596/659/1/012022/pdf
https://arxiv.org/pdf/2012.08202.pdf
"""


from ._calibration_measures import anees, nci
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
    "rmse",
    "relative_rmse",
    "mae",
    "relative_mae",
    "max_error",
    "relative_max_error",
    "mean_error",
    "relative_mean_error",
    "anees",
    "nci",
    "gaussianity_p_value",
    "sample_reference_distance",
    "sample_sample_distance",
]
