"""Work precision diagrams."""


class WorkPrecision:
    """Build-your-own work-precision diagram."""

    def __init__(self, algorithm):
        self.algorithm = algorithm

    def evaluate(self, problem):
        """Run the simulations and return a dict containing the collected info."""
        return {"problem": problem, "algorithm": self.algorithm}

    def collect_walltime(self):
        """Include wall-time in results."""
        # pylint: disable=unnecessary-pass
        pass

    def collect_rmse(self, evalgrid=None):
        """Include RMSE in results."""
        # pylint: disable=unnecessary-pass
        pass

    def collect_L2(self, evalgrid=None):
        """Include L2-error in results."""
        # pylint: disable=unnecessary-pass
        pass
