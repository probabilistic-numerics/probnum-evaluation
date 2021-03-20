"""Tests for calibration measure functions."""
import numpy as np
import probnum as pn
import pytest

from probnumeval.timeseries import (
    average_normalized_estimation_error_squared,
    chi2_confidence_intervals,
    non_credibility_index,
    non_credibility_index2,
    non_credibility_index3,
)


@pytest.fixture
def kalpost():
    """Kalman posterior with irrelevant values.

    Used to test calibration measures
    """
    rvlist = [
        pn.randvars.Normal(mean=i + np.arange(2, 4), cov=np.diag(np.arange(4, 6)))
        for i in range(10)
    ]
    locs = np.linspace(0.0, 1.0, 10)
    return pn.filtsmooth.FilteringPosterior(
        states=rvlist, locations=locs, transition=pn.statespace.IBM(1, 1)
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
    output = average_normalized_estimation_error_squared(kalpost, refsol, grid)

    assert np.isscalar(output)
    assert output > 0


def test_chi2_confidence():
    lower, upper = chi2_confidence_intervals(dim=2)

    assert lower == pytest.approx(0.01, rel=1e-1)
    assert upper == pytest.approx(10, rel=1e-1)


def test_nci():
    with pytest.raises(NotImplementedError):
        non_credibility_index(None, None, None)


def test_nci2():
    with pytest.raises(NotImplementedError):
        non_credibility_index2(None, None, None)


def test_nci3():
    with pytest.raises(NotImplementedError):
        non_credibility_index3(None, None, None)
