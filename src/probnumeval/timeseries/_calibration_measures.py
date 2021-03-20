"""Uncertainty calibration measures.

Source for ANEES, NCI, NCI2, NCI3:
http://folk.ntnu.no/skoge/prost/proceedings/ifac2002/data/content/02779/2779.pdf

Other source
https://iopscience.iop.org/article/10.1088/1742-6596/659/1/012022/pdf
"""

from typing import Callable

import numpy as np
import probnum as pn
import scipy.stats

__all__ = [
    "average_normalised_estimation_error_squared",
    "chi2_confidence_intervals",
    "non_credibility_index",
    "non_credibility_index2",
    "non_credibility_index3",
]


def average_normalised_estimation_error_squared(
    approximate_solution: pn.filtsmooth.TimeSeriesPosterior,
    reference_solution: Callable[[np.ndarray], np.ndarray],
    locations: np.ndarray,
):
    """Compute the average normalised estimation error squared.

    Also known as chi-squared statistic.

    Parameters
    ----------
    approximate_solution :
        Approximate solution as returned by a Kalman filter or ODE solver.
    reference_solution :
        Reference solution. (This is not assumed to be a `TimeSeriesPosterior`, because
        ideally this is the true solution of a problem; often, it is a reference solution
        computed with a non-probabilistic algorithm.)
    locations :
        Set of locations on which to evaluate the statistic.
    """
    approximate_evaluation = approximate_solution(locations)
    reference_evaluation = reference_solution(locations)
    cov_matrices = approximate_evaluation.cov
    centered_mean = approximate_evaluation.mean - reference_evaluation.mean

    intermediate = np.einsum("nd,ndd->nd", centered_mean, cov_matrices)
    final = np.einsum("nd,nd->n", intermediate, centered_mean)
    return final.mean(axis=0)


def chi2_confidence_intervals(dim, perc=0.99):
    """Easily access the confidence intervals of a chi-squared RV."""
    delta = (1.0 - perc) / 2.0
    lower = scipy.stats.chi2(df=dim).ppf(delta)
    upper = scipy.stats.chi2(df=dim).ppf(1 - delta)
    return lower, upper


def non_credibility_index(
    approximate_solution: pn.filtsmooth.TimeSeriesPosterior,
    reference_solution: Callable[[np.ndarray], np.ndarray],
    locations: np.ndarray,
):
    """Compute the non-credibility index."""
    raise NotImplementedError


def non_credibility_index2(
    approximate_solution: pn.filtsmooth.TimeSeriesPosterior,
    reference_solution: Callable[[np.ndarray], np.ndarray],
    locations: np.ndarray,
):
    """Compute a variant of the non-credibility index."""
    raise NotImplementedError


def non_credibility_index3(
    approximate_solution: pn.filtsmooth.TimeSeriesPosterior,
    reference_solution: Callable[[np.ndarray], np.ndarray],
    locations: np.ndarray,
):
    """Compute a variant of the non-credibility index."""
    raise NotImplementedError
