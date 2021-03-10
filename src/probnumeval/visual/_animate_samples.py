"""Animate samples from a standard Normal distribution."""

import numpy as np


def random_great_circle_of_unitsphere(
    d, num_steps, initial_sample=None, initial_direction=None
):
    """Draw a great circle on a unitsphere uniformly at random.

    Based on the MATLAB implementation in [1].

    This can be used to "make a sample from a GP move around in the sample space".
    In this case, d is the number of time-grid points
    (which amounts to the dimension of the underlying multivariate Normal distribution),
    and num_steps is the number of frames that shall be shown in the animation.

    Parameters
    ----------
    d :
        Dimension of the sphere. This can be thought of as the dimension of the
        underlying multivariate Normal distribution of which samples shall be animated.
    num_steps :
        Number of steps to be taken. This can be thought of the number of frames
        in the final animation.
    initial_sample:
        **Shape (d,).**
        Initial sample on the sphere. Will be normalized to length 1 internally. Optional.
        If not provided, sampled from a standard Normal distribution.
    initial_direction:
        **Shape (d,).**
        Initial direction on the tangent space of the initial sample. Will be orthonormalized internally.
        Optional. If not provided, sampled from a standard Normal distribution.

    Returns
    -------
    np.ndarray
        **Shape (d, num_steps).**
        N steps that traverse the sphere along a (d-1)-dimensional subspace.

    References
    ----------
    .. [1]
        Philipp Hennig. Animating Samples from Gaussian Distributions.
        Technical Report No. 8 of the Max Planck Institute for Intelligent Systems. September 2013

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> dim, num_steps = 2, 10
    >>> initial_sample = np.random.randn(2, 1)
    >>> initial_direction = np.random.randn(2, 1)
    >>> out = random_great_circle_of_unitsphere(dim, num_steps, initial_sample, initial_direction)
    >>> print(np.round(out, 1))
    [[ 0.5  0.5  0.3 -0.  -0.3 -0.5 -0.5 -0.3  0.   0.3]
     [-0.1  0.2  0.4  0.5  0.4  0.1 -0.2 -0.4 -0.5 -0.4]]
    """
    # Read inputs
    state = initial_sample if initial_sample is not None else np.random.randn(d, 1)
    direction = (
        initial_direction if initial_direction is not None else np.random.randn(d, 1)
    )

    # Normalize and orthogonalize
    scale = np.linalg.norm(state)
    normalized_state = state / scale
    orthogonal_direction = (
        direction - (direction.T @ normalized_state) * normalized_state
    )
    orthonormal_direction = orthogonal_direction / np.linalg.norm(orthogonal_direction)

    # Compute great circle
    equispaced_distances = np.linspace(0, 2.0 * np.pi, num_steps + 1)[:-1]
    out = np.array(
        [
            scale * _geodesic_sphere(normalized_state, orthonormal_direction * delta)
            for delta in equispaced_distances
        ]
    )
    return out[:, :, 0].T


def geodesic_sphere(point, velocity):
    r"""Compute the geodesic on the sphere.

    It is given by the exponential map starting at a point :math:`p` and initial velocity `v t`,

    .. math:: \gamma(t) := \text{Exp}_p(v t) = \cos(\|v\| t) p + \sin(\|v\| t) \frac{v}{\|v\|}

    and can be used to compute a great circle on a sphere.
    """

    # Decompose the velocity into magnitude * direction
    magnitude = np.linalg.norm(velocity)
    direction = velocity / magnitude

    # Early exit if no proper direction is given
    if magnitude == 0.0:
        return point

    geodesic = np.cos(magnitude) * point + np.sin(magnitude) * direction
    return geodesic
