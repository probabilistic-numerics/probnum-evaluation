"""Types."""

from typing import Callable

import numpy as np
from probnum import filtsmooth

__all__ = ["ProbabilisticSolutionType", "DeterministicSolutionType"]

ProbabilisticSolutionType = filtsmooth.TimeSeriesPosterior

DeterministicSolutionType = Callable[[np.ndarray], np.ndarray]
