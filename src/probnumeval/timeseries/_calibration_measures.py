"""Uncertainty calibration measures.

Source for ANEES, NCI, NCI2, NCI3:
http://folk.ntnu.no/skoge/prost/proceedings/ifac2002/data/content/02779/2779.pdf

Other source
https://iopscience.iop.org/article/10.1088/1742-6596/659/1/012022/pdf
"""

import scipy.stats


# aka chi2 test
def average_normalised_estimation_error_squared(sol, ref_sol, evalgrid):
    """Compute the average normalised estimation error squared.

    Also known as chi-squared statistic.
    """
    raise NotImplementedError


def chi2_confidence_intervals(dim, perc=0.99):
    """Easily access the confidence intervals of a chi-squared RV."""
    delta = (1.0 - perc) / 2.0
    lower = scipy.stats.chi2(df=dim).ppf(delta)
    upper = scipy.stats.chi2(df=dim).ppf(1 - delta)
    return lower, upper


def non_credibility_index(sol, ref_sol, evalgrid):
    """Compute the non-credibility index."""
    raise NotImplementedError


def non_credibility_index2(sol, ref_sol, evalgrid):
    """Compute a variant of the non-credibility index."""
    raise NotImplementedError


def non_credibility_index3(sol, ref_sol, evalgrid):
    """Compute a variant of the non-credibility index."""
    raise NotImplementedError
