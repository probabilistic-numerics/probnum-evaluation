"""Animate samples from a standard Normal distribution."""

import numpy as np


def animate_with_random_great_circle_of_unitsphere(
    d, num_steps, initial_sample=None, initial_direction=None, endpoint=False
):
    """Animate samples from a standard Normal distribution by drawing a great circle on
    a unitsphere uniformly at random.

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
    endpoint
        Whether the final state should be equal to the first state. Optional. Default is False.


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
    >>> states = animate_with_random_great_circle_of_unitsphere(dim, num_steps)
    >>> print(np.round(states, 1))
    [[ 0.5  0.5  0.3 -0.  -0.3 -0.5 -0.5 -0.3  0.   0.3]
     [-0.1  0.2  0.4  0.5  0.4  0.1 -0.2 -0.4 -0.5 -0.4]]
    """
    # Read inputs
    state = initial_sample if initial_sample is not None else np.random.randn(d)
    direction = (
        initial_direction if initial_direction is not None else np.random.randn(d)
    )

    # Normalize and orthogonalize
    scale = np.linalg.norm(state)
    normalized_state = state / scale
    orthogonal_direction = (
        direction - (direction.T @ normalized_state) * normalized_state
    )
    orthonormal_direction = orthogonal_direction / np.linalg.norm(orthogonal_direction)

    # Compute great circle
    equispaced_distances = np.linspace(0, 2.0 * np.pi, num_steps, endpoint=endpoint)
    untransposed_states = np.array(
        [
            scale * geodesic_sphere(normalized_state, orthonormal_direction * delta)
            for delta in equispaced_distances
        ]
    )
    return untransposed_states.T


def geodesic_sphere(point, velocity):
    r"""Compute the geodesic on the sphere.

    It is given by the exponential map starting at a point :math:`p` and initial velocity `v t`,

    .. math:: \gamma(t) := \text{Exp}_p(v t) = \cos(\|v\| t) p + \sin(\|v\| t) \frac{v}{\|v\|}

    and can be used to compute a great circle on a sphere.
    The dimension of the sphere is read off the sizes of point and velocity.
    """

    # Decompose the velocity into magnitude * direction
    magnitude = np.linalg.norm(velocity)
    direction = velocity / magnitude

    # Early exit if no proper direction is given
    if magnitude == 0.0:
        return point

    geodesic = np.cos(magnitude) * point + np.sin(magnitude) * direction
    return geodesic


def animate_with_periodic_gp(d, num_steps, base_measure_sample=None, endpoint=False):
    """Animate samples from a standard Normal distribution by drawing samples from a
    periodic Gaussian process.

    Parameters
    ----------
    d :
        Dimension of the underlying multivariate Normal distribution of which samples shall be animated.
    num_steps :
        Number of steps to be taken. This can be thought of the number of frames
        in the final animation.
    base_measure_sample:
        **Shape (d, num_steps).**
        I.i.d. samples from a standard Normal distribution.
    endpoint
        Whether the final state should be equal to the first state. Optional. Default is False.


    Returns
    -------
    np.ndarray
        **Shape (d, num_steps).**
        N steps that traverse the sphere along a (d-1)-dimensional subspace.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> dim, num_steps = 2, 10
    >>> states = animate_with_periodic_gp(dim, num_steps)
    >>> print(np.round(states, 1))
    [[ 0.5  0.3  0.6  1.6  1.1  0.5  0.3  0.6  1.6  1.1]
     [-0.5 -0.7 -0.3 -1.3 -1.8 -0.5 -0.7 -0.3 -1.3 -1.8]]
    """

    def k(t1, t2):
        return np.exp(-np.sin(np.abs(t1 - t2)) ** 2)

    unit_sample = (
        base_measure_sample
        if base_measure_sample is not None
        else np.random.randn(d, num_steps)
    )

    equispaced_distances = np.linspace(0, 2 * np.pi, num_steps, endpoint=endpoint)
    m = np.zeros(len(equispaced_distances))
    K = k(equispaced_distances[:, None], equispaced_distances[None, :])

    # Transform "from the right", because unit_sample is shape (d, num_steps)
    damping_factor = 1e-12
    KS = np.linalg.cholesky(K + damping_factor * np.eye(len(K)))
    samples = unit_sample @ KS.T + m[None, :]
    return samples
