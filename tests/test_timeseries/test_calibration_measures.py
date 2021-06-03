"""Tests for calibration measure functions."""
import numpy as np
import pytest
from probnum import filtsmooth, randvars, statespace

from probnumeval import config
from probnumeval.timeseries import (
    average_normalized_estimation_error_squared,
    chi2_confidence_intervals,
    non_credibility_index,
)


def test_chi2_confidence_intervals():
    lower, upper = chi2_confidence_intervals(dim=2)

    assert lower == pytest.approx(0.01, rel=1e-1)
    assert upper == pytest.approx(10, rel=1e-1)


@pytest.fixture
def kalpost():
    """Kalman posterior with irrelevant values.

    Used to test calibration measures
    """
    rvlist = [
        randvars.Normal(mean=i + np.arange(2, 4), cov=np.diag(np.arange(4, 6)))
        for i in range(10)
    ]
    locs = np.linspace(0.0, 1.0, 10)
    return filtsmooth.FilteringPosterior(
        states=rvlist, locations=locs, transition=statespace.IBM(1, 1)
    )


@pytest.fixture
def refsol():
    """Reference solution.

    Garbage values, but that does not matter.
    """
    return lambda x: np.random.rand(*(x.shape + (2,)))


@pytest.fixture
def grid():
    return np.linspace(0.0, 1.0, 15)


def test_anees(kalpost, refsol, grid):
    """The average normalized estimation error squared is a positive scalar."""
    with config.covariance_inversion_context(strategy="inv"):
        output = average_normalized_estimation_error_squared(kalpost, refsol, grid)

    assert np.isscalar(output)
    assert output > 0


def test_nci(kalpost, refsol, grid):
    with config.covariance_inversion_context(strategy="inv"):
        output = non_credibility_index(kalpost, refsol, grid)
    assert np.isscalar(output)
