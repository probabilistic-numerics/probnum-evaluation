"""Tests for sample analysis functions."""
import pytest

from probnumeval.timeseries import (
    average_intersample_rmse,
    average_rmse,
    gaussianity_p_value,
)


def test_average_intersample_rmse():
    with pytest.raises(NotImplementedError):
        average_intersample_rmse(None)


def test_average_rmse():
    with pytest.raises(NotImplementedError):
        average_rmse(None, None)


def test_normaltest():
    with pytest.raises(NotImplementedError):
        gaussianity_p_value(None)
