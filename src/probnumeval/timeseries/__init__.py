"""Work-precision diagrams and calibration analysis for time series problems.

Applicable to (O)DE solvers, filters, smoothers, and more.
"""


from ._error_measures import final_time_error, max_error, root_mean_square_error
from ._statistcs import chi2_statistic, nci_statistic

__all__ = [
    "root_mean_square_error",
    "max_error",
    "final_time_error",
    "chi2_statistic",
    "nci_statistic",
]
