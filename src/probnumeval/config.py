"""Configurations for all sorts of things."""

__all__ = ["covariance_inversion"]

covariance_inversion = dict(
    strategy="cholesky",
    symmetrize=True,
    damping=1e-14,
)
