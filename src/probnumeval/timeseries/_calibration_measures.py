"""Uncertainty calibration measures.

Source for ANEES, NCI, NCI2, NCI3:
http://folk.ntnu.no/skoge/prost/proceedings/ifac2002/data/content/02779/2779.pdf

Other source
https://iopscience.iop.org/article/10.1088/1742-6596/659/1/012022/pdf
"""

from typing import Callable

import numpy as np
import scipy.stats

from probnumeval.type import ApproximateSolutionType, ReferenceSolutionType

__all__ = [
    "average_normalized_estimation_error_squared",
    "non_credibility_index",
    "chi2_confidence_intervals",
]


def average_normalized_estimation_error_squared(
    approximate_solution: ApproximateSolutionType,
    reference_solution: ReferenceSolutionType,
    locations: np.ndarray,
):
    r"""Compute the average normalised estimation error squared.

    Also known as chi-squared statistic. It computes

    .. math:: \chi^2 :=
        \frac{1}{N + 1}
        \sum_{n=0}^N
        (y^*(t_n) - \mathbb{E}[y(t_n)])^\top
        \mathbb{C}[y(t_n)]^{-1}
        (y^*(t_n) - \mathbb{E}[y(t_n)])

    where :math:`\mathbb{E}` is the mean and :math:`\mathbb{C}` is the covariance.
    If :math:`y` is a Gaussian process, :math:`\chi^2` follows a chi-squared distribution.


    Parameters
    ----------
    approximate_solution :
        Approximate solution as returned by a Kalman filter or ODE solver. This must be a `FiltSmoothPosterior`.
    reference_solution :
        Reference solution. (This is not assumed to be a `TimeSeriesPosterior`, because
        ideally this is the true solution of a problem; often, it is a reference solution
        computed with a non-probabilistic algorithm.)
    locations :
        Set of locations on which to evaluate the statistic.

    Returns
    -------
    ANEES statistic (i.e. :math:`\chi^2` above).
    """
    centered_mean, cov_matrices = _compute_components(
        approximate_solution, locations, reference_solution
    )

    normalized_discrepancies = _compute_normalized_discrepancies(
        centered_mean, cov_matrices
    )
    return np.mean(normalized_discrepancies)


def non_credibility_index(
    approximate_solution: ApproximateSolutionType,
    reference_solution: ReferenceSolutionType,
    locations: np.ndarray,
):
    r"""Compute the non-credibility index (NCI).

    Parameters
    ----------
    approximate_solution :
        Approximate solution as returned by a Kalman filter or ODE solver. This must be a `FiltSmoothPosterior`.
    reference_solution :
        Reference solution. (This is not assumed to be a `TimeSeriesPosterior`, because
        ideally this is the true solution of a problem; often, it is a reference solution
        computed with a non-probabilistic algorithm.)
    locations :
        Set of locations on which to evaluate the statistic.

    Returns
    -------
    NCI statistic.
    """
    centered_mean, cov_matrices = _compute_components(
        approximate_solution, locations, reference_solution
    )
    sample_covariance_matrix = np.cov(centered_mean.T)
    normalized_discrepancies = _compute_normalized_discrepancies(
        centered_mean, cov_matrices
    )
    intermediate = centered_mean @ sample_covariance_matrix
    reference_discrepancies = np.einsum("nd,nd->n", intermediate, centered_mean)
    return 10 * (
        np.mean(np.log10(normalized_discrepancies))
        - np.mean(np.log10(reference_discrepancies))
    )


def _compute_components(approximate_solution, locations, reference_solution):
    approximate_evaluation = approximate_solution(locations)
    reference_evaluation = reference_solution(locations)
    cov_matrices = approximate_evaluation.cov
    centered_mean = approximate_evaluation.mean - reference_evaluation
    return centered_mean, cov_matrices


def _compute_normalized_discrepancies(centered_mean, cov_matrices):
    intermediate = np.einsum("nd,ndd->nd", centered_mean, cov_matrices)
    final = np.einsum("nd,nd->n", intermediate, centered_mean)
    return final


def chi2_confidence_intervals(dim, perc=0.99):
    """Easily access the confidence intervals of a chi-squared RV."""
    delta = (1.0 - perc) / 2.0
    lower = scipy.stats.chi2(df=dim).ppf(delta)
    upper = scipy.stats.chi2(df=dim).ppf(1 - delta)
    return lower, upper
