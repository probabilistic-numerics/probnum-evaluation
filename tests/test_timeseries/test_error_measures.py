"""Tests for error measure functions."""
import numpy as np
import pytest

from probnumeval import timeseries


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

    return np.linspace(0.1, 1.0)


def test_rmse(sol, ref_sol, evalgrid):
    assert 0.0 < timeseries.rmse(sol, ref_sol, evalgrid) < 1e-8


def test_mae(sol, ref_sol, evalgrid):
    assert 0.0 < timeseries.rmse(sol, ref_sol, evalgrid) < 1e-8


def test_max_error(sol, ref_sol, evalgrid):
    assert 0.0 < timeseries.max_error(sol, ref_sol, evalgrid) < 1e-8


def test_relative_rmse(sol, ref_sol, evalgrid):
    assert 0.0 < timeseries.relative_rmse(sol, ref_sol, evalgrid) < 1e-8


def test_relative_mae(sol, ref_sol, evalgrid):
    assert 0.0 < timeseries.relative_mae(sol, ref_sol, evalgrid) < 1e-8


def test_relative_max_error(sol, ref_sol, evalgrid):
    assert 0.0 < timeseries.relative_max_error(sol, ref_sol, evalgrid) < 1e-8
