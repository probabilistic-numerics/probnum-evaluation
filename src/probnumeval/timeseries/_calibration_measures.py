"""Uncertainty calibration measures.

Source for ANEES, NCI, NCI2, NCI3:
http://folk.ntnu.no/skoge/prost/proceedings/ifac2002/data/content/02779/2779.pdf

Other source
https://iopscience.iop.org/article/10.1088/1742-6596/659/1/012022/pdf
"""

import numpy as np

from probnumeval import multivariate
from probnumeval.type import DeterministicSolutionType, ProbabilisticSolutionType

__all__ = [
    "anees",
    "non_credibility_index",
    "inclination_index",
]


def anees(
    approximate_solution: ProbabilisticSolutionType,
    reference_solution: DeterministicSolutionType,
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
    For a :math:`d` dimensional solution, the outcome is

    - **Underconfident** if :math:`\chi^2 < d` holds. The estimated error is way larger than the actual error.
    - **Overconfident** if :math:`\chi^2 > d` holds. The estimated error is way smaller than the actual error.

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

    See also
    --------
    chi2_confidence_intervals
        Confidence intervals for the ANEES test statistic.
    nci
        An alternative calibration measure.

    """

    approximate_evaluation = approximate_solution(locations)
    reference_evaluation = reference_solution(locations)
    return multivariate.anees(
        approximate_solution=approximate_evaluation,
        reference_solution=reference_evaluation,
    )


def non_credibility_index(
    approximate_solution: ProbabilisticSolutionType,
    reference_solution: DeterministicSolutionType,
    locations: np.ndarray,
):
    r"""Compute the non-credibility index (NCI).

    The NCI indicates whether an estimate is

    - **Underconfident** if :math:`\text{NCI} < 0` holds. The estimated error is way larger than the actual error.
    - **Overconfident** if :math:`\text{NCI} > 0` holds. The estimated error is way smaller than the actual error.


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

    See also
    --------
    anees
        An alternative calibration measure.

    """
    approximate_evaluation = approximate_solution(locations)
    reference_evaluation = reference_solution(locations)
    return multivariate.non_credibility_index(
        approximate_solution=approximate_evaluation,
        reference_solution=reference_evaluation,
    )


def inclination_index(
    approximate_solution: ProbabilisticSolutionType,
    reference_solution: DeterministicSolutionType,
    locations: np.ndarray,
):
    r"""Compute the inclination index (II).

    The II indicates whether an estimate is

    - **Underconfident** if :math:`\text{NCI} < 0` holds. The estimated error is way larger than the actual error.
    - **Overconfident** if :math:`\text{NCI} > 0` holds. The estimated error is way smaller than the actual error.


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

    See also
    --------
    anees
        An alternative calibration measure.
    non_credibility_index
        An alternative calibration measure.
    """
    approximate_evaluation = approximate_solution(locations)
    reference_evaluation = reference_solution(locations)
    return multivariate.inclination_index(
        approximate_solution=approximate_evaluation,
        reference_solution=reference_evaluation,
    )
