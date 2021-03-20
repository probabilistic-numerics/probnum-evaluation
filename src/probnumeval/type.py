"""Types."""

from typing import Callable

import numpy as np
from probnum import filtsmooth

__all__ = ["ApproximateSolutionType", "ReferenceSolutionType"]

ApproximateSolutionType = filtsmooth.FiltSmoothPosterior
ReferenceSolutionType = Callable[[np.ndarray], np.ndarray]
