"""Test for calibration functions."""

import pytest

from probnumeval import utils


def test_chi2_confidence_intervals():
    lower, upper = utils.chi2_confidence_intervals(dim=2, perc=0.99)

    assert lower == pytest.approx(0.01, rel=1e-1)
    assert upper == pytest.approx(10, rel=1e-1)
