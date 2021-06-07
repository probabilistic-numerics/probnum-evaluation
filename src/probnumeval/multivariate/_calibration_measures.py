"""Uncertainty calibration measures."""

from typing import Union

import numpy as np
import scipy.linalg
import scipy.stats
from probnum import _randomvariablelist, randvars

from probnumeval import config

__all__ = [
    "anees",
    "non_credibility_index",
    "inclination_index",
]

# The following pylint-exception is for the _randomvariablelist access:
# pylint: disable=protected-access


def anees(
    approximate_solution: Union[
        randvars.Normal, _randomvariablelist._RandomVariableList
    ],
    reference_solution: np.ndarray,
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
    For a :math:`d` dimensional solution, the outcome is

    - **Underconfident** if :math:`\chi^2 < d` holds. The estimated error is way larger than the actual error.
    - **Overconfident** if :math:`\chi^2 > d` holds. The estimated error is way smaller than the actual error.

    Parameters
    ----------
    approximate_solution :
        Approximate solution as returned by a (Gaussian) probabilistic numerical method.
    reference_solution :
        Reference solution. This is an array, because it must be a deterministic point-estimate.

    Returns
    -------
    ANEES statistic (i.e. :math:`\chi^2` above).

    See also
    --------
    chi2_confidence_intervals
        Confidence intervals for the ANEES test statistic.
    non_credibility_index
        An alternative calibration measure.

    """
    centered_mean = approximate_solution.mean - reference_solution
    cov_matrices = approximate_solution.cov

    centered_mean = np.atleast_2d(centered_mean)
    cov_matrices = np.atleast_3d(cov_matrices)

    normalized_discrepancies = _compute_normalized_discrepancies(
        centered_mean, cov_matrices
    )
    return np.mean(normalized_discrepancies)


def non_credibility_index(
    approximate_solution: _randomvariablelist._RandomVariableList,
    reference_solution: np.ndarray,
):
    r"""Compute the non-credibility index (NCI).

    The NCI indicates how credible an estimate is. The smaller this value, the better. The NCI of a perfectly
    credible estimator is zero.
    Unlike the inclination index, the NCI cannot determine over- and underconfidence.

    Parameters
    ----------
    approximate_solution :
        Approximate solution as returned by a (Gaussian) probabilistic numerical method.
    reference_solution :
        Reference solution. This is an array, because it must be a deterministic point-estimate.

    Returns
    -------
    NCI statistic.

    See also
    --------
    anees
        An alternative calibration measure.
    inclination_index
        A version of the NCI that can figure out over- or underconfidence.

    """
    if not isinstance(approximate_solution, _randomvariablelist._RandomVariableList):
        raise TypeError(
            "The non-credibility index is only valid for a collection of random variables."
        )

    centered_mean = approximate_solution.mean - reference_solution
    cov_matrices = approximate_solution.cov

    centered_mean = np.atleast_2d(centered_mean)
    cov_matrices = np.atleast_3d(cov_matrices)

    normalized_discrepancies = _compute_normalized_discrepancies(
        centered_mean, cov_matrices
    )

    sample_covariance_matrix = np.tile(
        np.cov(centered_mean.T), reps=(len(centered_mean), 1, 1)
    )

    reference_discrepancies = _compute_normalized_discrepancies(
        centered_mean, sample_covariance_matrix
    )
    nci = 10 * (
        np.mean(
            np.abs(
                np.log10(normalized_discrepancies) - np.log10(reference_discrepancies)
            )
        )
    )
    return nci


def inclination_index(
    approximate_solution: Union[
        randvars.Normal, _randomvariablelist._RandomVariableList
    ],
    reference_solution: np.ndarray,
):
    r"""Compute the inclination index (II).

    The II is a version of the NCI that additionally indicates whether an estimate is

    - **Underconfident** if :math:`\text{II} < 0` holds. The estimated error is way larger than the actual error.
    - **Overconfident** if :math:`\text{II} > 0` holds. The estimated error is way smaller than the actual error.

    Parameters
    ----------
    approximate_solution :
        Approximate solution as returned by a (Gaussian) probabilistic numerical method.
    reference_solution :
        Reference solution. This is an array, because it must be a deterministic point-estimate.

    Returns
    -------
    Inclination index.

    See also
    --------
    anees
        An alternative calibration measure.
    non_credibility_index
        Non-credibility index.
    """
    if not isinstance(approximate_solution, _randomvariablelist._RandomVariableList):
        raise TypeError(
            "The inclination index is only valid for a collection of random variables."
        )

    cov_matrices = approximate_solution.cov
    centered_mean = approximate_solution.mean - reference_solution

    normalized_discrepancies = _compute_normalized_discrepancies(
        centered_mean, cov_matrices
    )

    sample_covariance_matrix = np.tile(
        np.cov(centered_mean.T), reps=(len(centered_mean), 1, 1)
    )

    reference_discrepancies = _compute_normalized_discrepancies(
        centered_mean, sample_covariance_matrix
    )
    ii = 10 * (
        np.mean(np.log10(normalized_discrepancies))
        - np.mean(np.log10(reference_discrepancies))
    )
    return ii


def _compute_normalized_discrepancies(centered_mean, cov_matrices):
    return np.array(
        [
            _compute_normalized_discrepancy(m, C)
            for (m, C) in zip(centered_mean, cov_matrices)
        ]
    )


def _compute_normalized_discrepancy(mean, cov):

    if config.COVARIANCE_INVERSION["symmetrize"]:
        cov = 0.5 * (cov + cov.T)
    if config.COVARIANCE_INVERSION["damping"] > 0.0:
        cov += config.COVARIANCE_INVERSION["damping"] * np.eye(len(cov))

    if config.COVARIANCE_INVERSION["strategy"] == "inv":
        return mean @ np.linalg.inv(cov) @ mean
    if config.COVARIANCE_INVERSION["strategy"] == "pinv":
        return mean @ np.linalg.pinv(cov) @ mean
    if config.COVARIANCE_INVERSION["strategy"] == "solve":
        return mean @ np.linalg.solve(cov, mean)
    if config.COVARIANCE_INVERSION["strategy"] == "cholesky":
        L, lower = scipy.linalg.cho_factor(cov, lower=True)
        return mean @ scipy.linalg.cho_solve((L, lower), mean)

    raise ValueError("Covariance inversion parameters are not known.")
