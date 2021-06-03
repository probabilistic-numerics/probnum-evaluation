"""Tests for configurations."""

import pytest

from probnumeval import config


def test_cov_inversion():
    cov_keys = config.COVARIANCE_INVERSION.keys()
    assert list(cov_keys) == ["strategy", "symmetrize", "damping"]


def test_cov_inversion_defaults():
    assert config.COVARIANCE_INVERSION["strategy"] == "cholesky"
    assert config.COVARIANCE_INVERSION["symmetrize"] == True
    assert config.COVARIANCE_INVERSION["damping"] == 1e-14


def test_context():
    assert config.COVARIANCE_INVERSION["strategy"] == "cholesky"
    assert config.COVARIANCE_INVERSION["symmetrize"] == True
    assert config.COVARIANCE_INVERSION["damping"] == 1e-14

    with config.covariance_inversion_context(
        strategy="pinv", symmetrize=True, damping=0.0
    ):
        assert config.COVARIANCE_INVERSION["strategy"] == "pinv"
        assert config.COVARIANCE_INVERSION["symmetrize"] == True
        assert config.COVARIANCE_INVERSION["damping"] == 0.0

    assert config.COVARIANCE_INVERSION["strategy"] == "cholesky"
    assert config.COVARIANCE_INVERSION["symmetrize"] == True
    assert config.COVARIANCE_INVERSION["damping"] == 1e-14
