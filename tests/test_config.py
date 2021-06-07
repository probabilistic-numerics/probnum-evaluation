"""Tests for configurations."""

from probnumeval import config


def test_cov_inversion():
    """Assert that the keys have not changed."""
    cov_keys = config.COVARIANCE_INVERSION.keys()
    assert list(cov_keys) == ["strategy", "symmetrize", "damping"]


def test_cov_inversion_defaults():
    """Check default values."""
    assert config.COVARIANCE_INVERSION["strategy"] == "cholesky"
    assert config.COVARIANCE_INVERSION["symmetrize"]
    assert config.COVARIANCE_INVERSION["damping"] == 0.0


def test_setter():
    """Check whether the setter function performs as expected."""
    assert config.COVARIANCE_INVERSION["strategy"] == "cholesky"
    assert config.COVARIANCE_INVERSION["symmetrize"]
    assert config.COVARIANCE_INVERSION["damping"] == 0.0

    config.set_covariance_inversion_parameters(
        strategy="inv", symmetrize=False, damping=10.0
    )

    assert config.COVARIANCE_INVERSION["strategy"] == "inv"
    assert not config.COVARIANCE_INVERSION["symmetrize"]
    assert config.COVARIANCE_INVERSION["damping"] == 10.0

    # Set back to previous values in order not to mess up the tests below.
    config.set_covariance_inversion_parameters(
        strategy="cholesky", symmetrize=True, damping=0.0
    )


def test_context():
    """Check whether the context manager does its job."""
    assert config.COVARIANCE_INVERSION["strategy"] == "cholesky"
    assert config.COVARIANCE_INVERSION["symmetrize"]
    assert config.COVARIANCE_INVERSION["damping"] == 0.0

    with config.covariance_inversion_context(
        strategy="pinv", symmetrize=False, damping=10.0
    ):
        assert config.COVARIANCE_INVERSION["strategy"] == "pinv"
        assert not config.COVARIANCE_INVERSION["symmetrize"]
        assert config.COVARIANCE_INVERSION["damping"] == 10.0

    assert config.COVARIANCE_INVERSION["strategy"] == "cholesky"
    assert config.COVARIANCE_INVERSION["symmetrize"]
    assert config.COVARIANCE_INVERSION["damping"] == 0.0
