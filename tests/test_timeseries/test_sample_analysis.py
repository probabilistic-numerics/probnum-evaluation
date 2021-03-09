"""Tests for sample analysis functions."""
import numpy as np
import pytest

from probnumeval.timeseries import (
    gaussianity_p_value,
    sample_reference_distance,
    sample_sample_distance,
)


@pytest.fixture
def fake_samples():
    return np.random.rand(100, 3)


@pytest.fixture
def fake_reference():
    return np.random.rand(3)


@pytest.mark.parametrize("p", [1, 2, np.inf])
def test_sample_sample_distance(fake_samples, p):
    """bla."""
    ssdist = sample_sample_distance(fake_samples, p=p)
    np.testing.assert_allclose(ssdist.shape, (100,))


@pytest.mark.parametrize("p", [1, 2, np.inf])
def test_sample_reference_distance(fake_samples, fake_reference, p):
    srdist = sample_reference_distance(fake_samples, fake_reference, p=p)
    np.testing.assert_allclose(srdist.shape, (100,))


def test_gaussianity_p_value():
    with pytest.raises(NotImplementedError):
        gaussianity_p_value(None)
