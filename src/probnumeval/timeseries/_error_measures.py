"""Error measures for time-series problems."""

import numpy as np

from probnumeval.type import DeterministicSolutionType

__all__ = ["root_mean_square_error", "final_time_error", "max_error"]


def root_mean_square_error(
    approximate_solution: DeterministicSolutionType,
    reference_solution: DeterministicSolutionType,
    locations: np.ndarray,
):
    """Compute an approximation of the L2 error on some grid."""
    approx_sol = approximate_solution(locations)
    ref_sol = reference_solution(locations)
    return np.linalg.norm(approx_sol - ref_sol) / np.sqrt(ref_sol.size)


def final_time_error(
    approximate_solution: DeterministicSolutionType,
    reference_solution: DeterministicSolutionType,
    locations: np.ndarray,
):
    """Compute the accumulated error."""
    out = approximate_solution(locations[-1])
    ref = reference_solution(locations[-1])
    return np.linalg.norm(out - ref) / np.sqrt(ref.size)


def max_error(
    approximate_solution: DeterministicSolutionType,
    reference_solution: DeterministicSolutionType,
    locations: np.ndarray,
):
    """Compute an approximation of the L-infinity error on some grid."""
    raise NotImplementedError
