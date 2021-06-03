"""Tests for configurations."""

import pytest

from probnumeval import config


def test_cov_inversion():
    cov_keys = config.covariance_inversion.keys()
    assert list(cov_keys) == ["strategy", "symmetrize", "damping"]


def test_cov_inversion_defaults():
    assert config.covariance_inversion["strategy"] == "cholesky"
    assert config.covariance_inversion["symmetrize"] == True
    assert config.covariance_inversion["damping"] == 1e-14
