"""Error measures."""

import numpy as np

__all__ = [
    "rmse",
    "relative_rmse",
    "mae",
    "relative_mae",
    "mean_error",
    "relative_mean_error",
]


def rmse(
    approximate_solution: np.ndarray,
    reference_solution: np.ndarray,
):
    """Compute the root mean-square error."""
    return mean_error(
        approximate_solution=approximate_solution,
        reference_solution=reference_solution,
        ord=2,
    )


def relative_rmse(
    approximate_solution: np.ndarray,
    reference_solution: np.ndarray,
):
    """Compute the root mean-square error."""
    return relative_mean_error(
        approximate_solution=approximate_solution,
        reference_solution=reference_solution,
        ord=2,
    )


def mae(
    approximate_solution: np.ndarray,
    reference_solution: np.ndarray,
):
    """Compute the root mean-square error."""
    return mean_error(
        approximate_solution=approximate_solution,
        reference_solution=reference_solution,
        ord=1,
    )


def relative_mae(
    approximate_solution: np.ndarray,
    reference_solution: np.ndarray,
):
    """Compute the root mean-square error."""
    return relative_mean_error(
        approximate_solution=approximate_solution,
        reference_solution=reference_solution,
        ord=1,
    )


def mean_error(
    approximate_solution: np.ndarray, reference_solution: np.ndarray, ord: int
):
    """Compute the mean error."""
    diff = (approximate_solution - reference_solution).flatten()
    normalization = reference_solution.size ** (1.0 / ord)
    return np.linalg.norm(diff, ord=ord) / normalization


def relative_mean_error(
    approximate_solution: np.ndarray, reference_solution: np.ndarray, ord: int
):
    """Compute the relative mean error."""
    diff = (approximate_solution - reference_solution) / reference_solution
    flat_diff = diff.flatten()
    normalization = reference_solution.size ** (1.0 / ord)
    return np.linalg.norm(flat_diff, ord=ord) / normalization
