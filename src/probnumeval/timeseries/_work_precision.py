"""Work precision diagrams."""

import time

import numpy as np


class WorkPrecision:
    """Build your own work-precision diagram.

    Examples
    --------

    Set up an example simulation: solve the logistic equation with scipy.integrate.solve_ivp

    >>> from numpy import round, array, linspace
    >>> from scipy.integrate import solve_ivp

    >>> def algo(rtol):
    ...     sol =  solve_ivp(lambda t, y: y*(1-y), (0., 2.), [0.1], rtol=rtol, dense_output=True)
    ...     return sol.sol

    >>> sol = algo(1e-3)
    >>> print(sol(0.5))
    [0.15481056]

    Compute a reference solution.

    >>> ref_sol = algo(1e-12)
    >>> print(ref_sol(0.5))
    [0.15482808]



    >>> wp = WorkPrecision(algorithm=algo, ref_sol=ref_sol, evalgrid=linspace(0., 2.))
    >>> results = wp.evaluate([1e-3, 1e-5, 1e-7, 1e-9], "rtol")
    >>> print(results["rtol"])
    [0.001, 1e-05, 1e-07, 1e-09]
    >>> print(*('{:.1e}'.format(x) for x in results["rmse"]), sep=' ')
    2.5e-05 7.2e-07 1.3e-08 1.3e-10
    >>> print(*('{:.1e}'.format(x) for x in results["final_time"]), sep=' ')
    4.1e-05 1.3e-06 1.4e-08 1.4e-10
    """

    def __init__(self, algorithm, ref_sol, evalgrid):

        self.algorithm = algorithm
        self.ref_sol = ref_sol
        self.evalgrid = evalgrid

    def evaluate(self, parameters, parameter_key):
        """Run the simulations and return a dict containing the collected info."""

        results = {parameter_key: [], "wall_time_sec": [], "rmse": [], "final_time": []}

        for par in parameters:
            results[parameter_key].append(par)

            start_time = time.time()
            sol = self.algorithm(par)
            end_time = time.time()
            results["wall_time_sec"].append(end_time - start_time)

            results["rmse"].append(compute_rmse(sol, self.ref_sol, self.evalgrid))
            results["final_time"].append(
                compute_final_time(sol, self.ref_sol, self.evalgrid)
            )

        return results


def compute_rmse(sol, ref_sol, evalgrid):
    out = sol(evalgrid)
    ref = ref_sol(evalgrid)
    return np.linalg.norm(out - ref) / np.sqrt(ref.size)


def compute_final_time(sol, ref_sol, evalgrid):
    out = sol(evalgrid[-1])
    ref = ref_sol(evalgrid[-1])
    return np.linalg.norm(out - ref) / np.sqrt(ref.size)
