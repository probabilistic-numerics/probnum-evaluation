"""Error measures for time-series problems."""

import numpy as np

from probnumeval import multivariate
from probnumeval.type import DeterministicSolutionType

__all__ = [
    "rmse",
    "relative_rmse",
    "mae",
    "relative_mae",
    "max_error",
    "relative_max_error",
    "mean_error",
    "relative_mean_error",
]


def rmse(
    approximate_solution: DeterministicSolutionType,
    reference_solution: DeterministicSolutionType,
    locations: np.ndarray,
):
    """Compute the root mean-square error."""
    return mean_error(
        approximate_solution=approximate_solution,
        reference_solution=reference_solution,
        locations=locations,
        p=2,
    )


def relative_rmse(
    approximate_solution: DeterministicSolutionType,
    reference_solution: DeterministicSolutionType,
    locations: np.ndarray,
):
    """Compute the root mean-square error."""
    return relative_mean_error(
        approximate_solution=approximate_solution,
        reference_solution=reference_solution,
        locations=locations,
        p=2,
    )


def max_error(
    approximate_solution: DeterministicSolutionType,
    reference_solution: DeterministicSolutionType,
    locations: np.ndarray,
):
    """Compute the root mean-square error."""
    return mean_error(
        approximate_solution=approximate_solution,
        reference_solution=reference_solution,
        locations=locations,
        p=np.inf,
    )


def relative_max_error(
    approximate_solution: DeterministicSolutionType,
    reference_solution: DeterministicSolutionType,
    locations: np.ndarray,
):
    """Compute the root mean-square error."""
    return relative_mean_error(
        approximate_solution=approximate_solution,
        reference_solution=reference_solution,
        locations=locations,
        p=np.inf,
    )


def mae(
    approximate_solution: DeterministicSolutionType,
    reference_solution: DeterministicSolutionType,
    locations: np.ndarray,
):
    """Compute the root mean-square error."""
    return mean_error(
        approximate_solution=approximate_solution,
        reference_solution=reference_solution,
        locations=locations,
        p=1,
    )


def relative_mae(
    approximate_solution: DeterministicSolutionType,
    reference_solution: DeterministicSolutionType,
    locations: np.ndarray,
):
    """Compute the root mean-square error."""
    return relative_mean_error(
        approximate_solution=approximate_solution,
        reference_solution=reference_solution,
        locations=locations,
        p=1,
    )


def mean_error(
    approximate_solution: DeterministicSolutionType,
    reference_solution: DeterministicSolutionType,
    locations: np.ndarray,
    p: int,
):
    """Compute the mean error."""
    approximate_evaluation = approximate_solution(locations)
    reference_evaluation = reference_solution(locations)
    return multivariate.mean_error(
        approximate_solution=approximate_evaluation,
        reference_solution=reference_evaluation,
        p=p,
    )


def relative_mean_error(
    approximate_solution: DeterministicSolutionType,
    reference_solution: DeterministicSolutionType,
    locations: np.ndarray,
    p: int,
):
    """Compute the relative mean error."""
    approximate_evaluation = approximate_solution(locations)
    reference_evaluation = reference_solution(locations)
    return multivariate.relative_mean_error(
        approximate_solution=approximate_evaluation,
        reference_solution=reference_evaluation,
        p=p,
    )
