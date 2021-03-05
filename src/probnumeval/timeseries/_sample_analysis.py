"""Extract information out of a bunch of samples from a solution."""

__all__ = ["average_rmse", "average_intersample_rmse", "gaussianity_p_value"]


def average_intersample_rmse(samples):
    """Compute the average intersample RMSE."""
    raise NotImplementedError


def average_rmse(samples, reference):
    """Compute the average RMSE."""
    raise NotImplementedError


def gaussianity_p_value(samples):
    """Compute a p-value that describes how closely a set of samples resembles samples
    from a Gaussian process."""
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html#scipy.stats.normaltest
    raise NotImplementedError
