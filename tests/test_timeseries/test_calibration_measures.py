"""Tests for calibration measure functions."""
import pytest

from probnumeval.timeseries import (
    average_normalised_estimation_error_squared,
    chi2_confidence_intervals,
    non_credibility_index,
    non_credibility_index2,
    non_credibility_index3,
)


def test_anees():
    with pytest.raises(NotImplementedError):
        average_normalised_estimation_error_squared(None, None, None)


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
