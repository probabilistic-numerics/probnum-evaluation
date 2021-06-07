"""Types."""

from typing import Callable

import numpy as np
from probnum import filtsmooth

__all__ = ["ProbabilisticSolutionType", "DeterministicSolutionType"]

ProbabilisticSolutionType = filtsmooth.TimeSeriesPosterior
"""Bla1."""

DeterministicSolutionType = Callable[[np.ndarray], np.ndarray]
"""Bla1."""
