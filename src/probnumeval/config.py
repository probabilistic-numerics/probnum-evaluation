"""Configurations for all sorts of things."""
from dataclasses import dataclass

__all__ = [
    "COVARIANCE_INVERSION",
    "covariance_inversion_context",
    "set_covariance_inversion_parameters",
]

COVARIANCE_INVERSION = dict(
    strategy="cholesky",
    symmetrize=True,
    damping=0.0,
)
"""Strategy parameters related to computing the inverse of a covariance matrix."""


def set_covariance_inversion_parameters(strategy, symmetrize, damping):
    """Change parameters of covariance inversion."""

    global COVARIANCE_INVERSION
    COVARIANCE_INVERSION = dict(
        strategy=strategy,
        symmetrize=symmetrize,
        damping=damping,
    )


@dataclass
class covariance_inversion_context:
    """Context manager for specific parameters of covariance inversion."""

    strategy: str
    symmetrize: bool = COVARIANCE_INVERSION["symmetrize"]
    damping: float = COVARIANCE_INVERSION["damping"]

    _old_values: dict = None

    def __enter__(self):
        self._old_values = COVARIANCE_INVERSION.copy()
        set_covariance_inversion_parameters(
            strategy=self.strategy,
            symmetrize=self.symmetrize,
            damping=self.damping,
        )

    def __exit__(self, type, value, traceback):
        set_covariance_inversion_parameters(
            strategy=self._old_values["strategy"],
            symmetrize=self._old_values["symmetrize"],
            damping=self._old_values["damping"],
        )
