"""Tests for error measure functions."""
import numpy as np
import pytest

from probnumeval.timeseries import final_time_error, max_error, root_mean_square_error


@pytest.fixture
def sol():
    def _sol(t):
        return t + 1e-10

    return _sol


@pytest.fixture
def ref_sol():
    def _ref_sol(t):
        return t

    return _ref_sol


@pytest.fixture
def evalgrid():

    return np.linspace(0.0, 1.0)


def test_rmse(sol, ref_sol, evalgrid):
    assert 0.0 < root_mean_square_error(sol, ref_sol, evalgrid) < 1e-8


def test_final_time_error(sol, ref_sol, evalgrid):
    assert 0.0 < final_time_error(sol, ref_sol, evalgrid) < 1e-8


def test_max_error(sol, ref_sol, evalgrid):
    with pytest.raises(NotImplementedError):
        max_error(sol, ref_sol, evalgrid)
