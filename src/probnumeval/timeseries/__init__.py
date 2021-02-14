"""Work-precision diagrams and calibration analysis for time series problems.

Applicable to (O)DE solvers, filters, smoothers, and more.

Some sources:
http://folk.ntnu.no/skoge/prost/proceedings/ifac2002/data/content/02779/2779.pdf
https://iopscience.iop.org/article/10.1088/1742-6596/659/1/012022/pdf
https://arxiv.org/pdf/2012.08202.pdf
"""


from ._calibration_measures import (
    average_normalised_estimation_error_squared,
    chi2_confidence_intervals,
    non_credibility_index,
    non_credibility_index2,
    non_credibility_index3,
)
from ._error_measures import final_time_error, max_error, root_mean_square_error
from ._sample_analysis import intersample_rmse, normaltest

__all__ = [
    "root_mean_square_error",
    "max_error",
    "final_time_error",
    "average_normalised_estimation_error_squared",
    "chi2_confidence_intervals",
    "non_credibility_index",
    "non_credibility_index2",
    "non_credibility_index3",
    "normaltest",
    "intersample_rmse",
]
