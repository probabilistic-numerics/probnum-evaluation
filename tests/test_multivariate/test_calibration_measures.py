"""Tests for calibration measure functions."""
import numpy as np
import pytest
from probnum import _randomvariablelist, randvars

from probnumeval import config, multivariate

all_strategies = pytest.mark.parametrize(
    "strategy", ["inv", "pinv", "solve", "cholesky"]
)
all_symmetries = pytest.mark.parametrize("symmetrize", [True, False])
all_dampings = pytest.mark.parametrize("damping", [1.0, 0.0])


# The following pylint-exception is for the _randomvariablelist access:
# pylint: disable=protected-access


@pytest.fixture
def approximate_solution():
    """List of normals with irrelevant values."""

    rvlist = [
        randvars.Normal(mean=i + np.arange(2, 4), cov=np.diag(np.arange(4, 6)))
        for i in range(10)
    ]

    return _randomvariablelist._RandomVariableList(rvlist)


@pytest.fixture
def reference_solution():
    """Reference solution.

    Garbage values, but that does not matter.
    """
    return np.random.rand(10, 2)


@all_strategies
@all_symmetries
@all_dampings
def test_anees(approximate_solution, reference_solution, strategy, symmetrize, damping):
    """The average normalized estimation error squared is a positive scalar."""
    with config.covariance_inversion_context(
        strategy=strategy, symmetrize=symmetrize, damping=damping
    ):
        output = multivariate.anees(approximate_solution, reference_solution)

    assert np.isscalar(output)
    assert output > 0


@all_strategies
@all_symmetries
@all_dampings
def test_nci(approximate_solution, reference_solution, strategy, symmetrize, damping):
    with config.covariance_inversion_context(
        strategy=strategy, symmetrize=symmetrize, damping=damping
    ):
        output = multivariate.nci(approximate_solution, reference_solution)
    assert np.isscalar(output)


@all_strategies
@all_symmetries
@all_dampings
def test_inclination_index(
    approximate_solution, reference_solution, strategy, symmetrize, damping
):
    with config.covariance_inversion_context(
        strategy=strategy, symmetrize=symmetrize, damping=damping
    ):
        output = multivariate.inclination_index(
            approximate_solution, reference_solution
        )
    assert np.isscalar(output)
