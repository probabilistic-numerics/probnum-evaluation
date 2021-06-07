"""Tests for error measure functions."""


import numpy as np
import pytest

from probnumeval import multivariate

all_shapes = pytest.mark.parametrize("shape", [1, (1,), (1, 1), 3, (3,), (3, 3)])


@pytest.fixture
def approximate_solution(shape):
    return 1.1 * np.ones(shape)


@pytest.fixture
def reference_solution(shape):
    return np.ones(shape)


@all_shapes
def test_mae(approximate_solution, reference_solution):
    mae = multivariate.mae(approximate_solution, reference_solution)
    assert np.isscalar(mae)
    assert mae == pytest.approx(0.1, rel=1e-1)


@all_shapes
def test_relative_mae(approximate_solution, reference_solution):
    relative_mae = multivariate.relative_mae(approximate_solution, reference_solution)
    assert np.isscalar(relative_mae)
    assert relative_mae == pytest.approx(0.1, rel=1e-1)


@all_shapes
def test_rmse(approximate_solution, reference_solution):
    rmse = multivariate.rmse(approximate_solution, reference_solution)
    assert np.isscalar(rmse)
    assert rmse == pytest.approx(0.1, rel=1e-1)


@all_shapes
def test_relative_mae(approximate_solution, reference_solution):
    relative_rmse = multivariate.relative_rmse(approximate_solution, reference_solution)
    assert np.isscalar(relative_rmse)
    assert relative_rmse == pytest.approx(0.1, rel=1e-1)
