"""Extract information out of a bunch of samples from a solution."""


import numpy as np
import scipy.spatial

__all__ = [
    "sample_reference_distance",
    "sample_sample_distance",
    "gaussianity_p_value",
]


def sample_sample_distance(samples: np.ndarray, p: int = 2) -> np.ndarray:
    r"""Compute the sample-sample distance.


    For a set of samples :math:`x_1, ..., x_N \in \mathbb{R}^d`, compute the set of dimension-normalized sample-sample distances :math:`E=(E_1, ..., E_N)` given by

    .. math:: E_k = \frac{1}{dN} \sum_{n=1}^N \| x_k - x_n \|_p

    for :math:`1 \leq p \leq \infty`. For :math:`p=2`, the root mean-squared error is recovered.

    Parameters
    ----------
    samples :
        **Shape (N, d).** Samples from the solution, evaluated at an end point.
    p :
        Order of the underlying norm that shall be used. At least 1, at most infinity. Default is 2, which corresponds to the RMSE.

    Returns
    -------
    np.ndarray
        **Shape (N,).** Sample-sample distances :math:`E=(E_1, ..., E_N)`.

    Examples
    --------
    >>> import numpy as np
    >>> fake_samples = np.arange(0, 300).reshape((100, 3))
    >>> rmse = sample_sample_distance(fake_samples, p=2)
    >>> print(rmse.shape)
    (100,)
    >>> print(np.round(np.mean(rmse), 1))
    57.7
    """
    distmat = scipy.spatial.distance_matrix(samples, samples, p=p) / samples.shape[1]
    return np.mean(distmat, axis=0)


def sample_reference_distance(
    samples: np.ndarray, reference: np.ndarray, p: int = 2
) -> np.ndarray:
    r"""Compute the sample-reference distance.

    For a set of samples :math:`x_1, ..., x_N \in \mathbb{R}^d` and reference solution :math:`\xi \in \mathbb{R}^d`, compute the set of dimension-normalized sample-reference distances :math:`R=(R_1, ..., R_N)` given by

    .. math:: R_k = \frac{1}{d} \| x_k - \xi \|_p

    for :math:`1 \leq p \leq \infty`. For :math:`p=2`, the root mean-squared error is recovered.

    Parameters
    ----------
    samples :
        **Shape (N, d).** Samples from the solution, evaluated at an end point.
    reference :
        **Shape (d,).** Reference solution..
    p :
        Order of the underlying norm that shall be used. At least 1, at most infinity. Default is 2, which corresponds to the RMSE.

    Returns
    -------
    np.ndarray
        **Shape (N,).** Sample-reference distances :math:`R=(R_1, ..., R_N)`.

    Examples
    --------
    >>> import numpy as np
    >>> fake_samples = np.arange(0, 300).reshape((100, 3))
    >>> fake_reference = np.arange(10, 13)
    >>> rmse = sample_reference_distance(fake_samples, fake_reference, p=2)
    >>> print(rmse.shape)
    (100,)
    >>> print(np.round(np.mean(rmse), 1))
    80.2
    """
    distmat = (
        scipy.spatial.distance_matrix(samples, reference[None, :], p=p)
        / samples.shape[1]
    )
    return distmat.flatten()


def gaussianity_p_value(samples):
    """Compute a p-value that describes how closely a set of samples resembles samples
    from a Gaussian process."""
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html#scipy.stats.normaltest
    raise NotImplementedError
