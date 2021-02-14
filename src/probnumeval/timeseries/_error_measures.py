"""Error measures for time-series problems."""


def root_mean_square_error(sol, ref_sol, evalgrid):
    """Compute an approximation of the L2 error on some grid."""
    out = sol(evalgrid)
    ref = ref_sol(evalgrid)
    return np.linalg.norm(out - ref) / np.sqrt(ref.size)


def final_time_error(sol, ref_sol, evalgrid):
    """Compute the accumulated error."""
    out = sol(evalgrid[-1])
    ref = ref_sol(evalgrid[-1])
    return np.linalg.norm(out - ref) / np.sqrt(ref.size)


def max_error(sol, ref_sol, evalgrid):
    """Compute an approximation of the L-infinity error on some grid."""
    raise NotImplementedError
