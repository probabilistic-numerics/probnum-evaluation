"""Configurations for all sorts of things."""
from dataclasses import dataclass

__all__ = ["covariance_inversion", "precision_context"]

covariance_inversion = dict(
    strategy="cholesky",
    symmetrize=True,
    damping=1e-14,
)


@dataclass
class precision_context:

    strategy: str
    symmetrize: bool
    damping: float

    _old_values: dict = None

    def __enter__(self):
        global covariance_inversion
        self._old_values = covariance_inversion
        covariance_inversion = dict(
            strategy=self.strategy,
            symmetrize=self.symmetrize,
            damping=self.damping,
        )

    def __exit__(self, type, value, traceback):
        global covariance_inversion
        covariance_inversion = self._old_values
