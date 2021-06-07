"""Error measures."""

import numpy as np

__all__ = [
    "rmse",
    "relative_rmse",
    "mae",
    "relative_mae",
    "mean_error",
    "relative_mean_error",
    "max_error",
    "relative_max_error",
]


def rmse(
    approximate_solution: np.ndarray,
    reference_solution: np.ndarray,
):
    """Compute the root mean-square error."""
    return mean_error(
        approximate_solution=approximate_solution,
        reference_solution=reference_solution,
        p=2,
    )


def relative_rmse(
    approximate_solution: np.ndarray,
    reference_solution: np.ndarray,
):
    """Compute the relative root mean-square error."""
    return relative_mean_error(
        approximate_solution=approximate_solution,
        reference_solution=reference_solution,
        p=2,
    )


def max_error(
    approximate_solution: np.ndarray,
    reference_solution: np.ndarray,
):
    """Compute the maximum error."""
    return mean_error(
        approximate_solution=approximate_solution,
        reference_solution=reference_solution,
        p=np.inf,
    )


def relative_max_error(
    approximate_solution: np.ndarray,
    reference_solution: np.ndarray,
):
    """Compute the relative maximum error."""
    return relative_mean_error(
        approximate_solution=approximate_solution,
        reference_solution=reference_solution,
        p=np.inf,
    )


def mae(
    approximate_solution: np.ndarray,
    reference_solution: np.ndarray,
):
    """Compute the mean absolute error."""
    return mean_error(
        approximate_solution=approximate_solution,
        reference_solution=reference_solution,
        p=1,
    )


def relative_mae(
    approximate_solution: np.ndarray,
    reference_solution: np.ndarray,
):
    """Compute the relative mean absolute error."""
    return relative_mean_error(
        approximate_solution=approximate_solution,
        reference_solution=reference_solution,
        p=1,
    )


def mean_error(
    approximate_solution: np.ndarray, reference_solution: np.ndarray, p: int
):
    """Compute the mean error."""
    approximate_solution = np.asarray(approximate_solution)
    reference_solution = np.asarray(reference_solution)

    diff = (approximate_solution - reference_solution).flatten()
    error = np.linalg.norm(diff, ord=p)

    if np.isinf(p):
        return error
    normalization = reference_solution.size ** (1.0 / p)
    return error / normalization


def relative_mean_error(
    approximate_solution: np.ndarray, reference_solution: np.ndarray, p: int
):
    """Compute the relative mean error."""
    approximate_solution = np.asarray(approximate_solution)
    reference_solution = np.asarray(reference_solution)

    diff = (approximate_solution - reference_solution) / reference_solution
    flat_diff = diff.flatten()
    error = np.linalg.norm(flat_diff, ord=p)

    if np.isinf(p):
        return error

    normalization = reference_solution.size ** (1.0 / p)
    return error / normalization
