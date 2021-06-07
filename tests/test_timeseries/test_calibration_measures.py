"""Tests for calibration measure functions."""
import numpy as np
import pytest
from probnum import filtsmooth, randvars, statespace

from probnumeval import config, timeseries

all_strategies = pytest.mark.parametrize(
    "strategy", ["inv", "pinv", "solve", "cholesky"]
)
all_symmetries = pytest.mark.parametrize("symmetrize", [True, False])
all_dampings = pytest.mark.parametrize("damping", [1.0, 0.0])


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


# pylint: disable=too-many-arguments


@all_strategies
@all_symmetries
@all_dampings
def test_anees(kalpost, refsol, grid, strategy, symmetrize, damping):
    """The average normalized estimation error squared is a positive scalar."""
    with config.covariance_inversion_context(
        strategy=strategy, symmetrize=symmetrize, damping=damping
    ):
        output = timeseries.anees(kalpost, refsol, grid)

    assert np.isscalar(output)
    assert output > 0


@all_strategies
@all_symmetries
@all_dampings
def test_nci(kalpost, refsol, grid, strategy, symmetrize, damping):
    with config.covariance_inversion_context(
        strategy=strategy, symmetrize=symmetrize, damping=damping
    ):
        output = timeseries.nci(kalpost, refsol, grid)
    assert np.isscalar(output)
