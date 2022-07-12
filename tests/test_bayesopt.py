import unittest

import numpy as np
import ConfigSpace.hyperparameters as CSH
from sklearn.gaussian_process.kernels import Matern

from bayesopt.bayesopt import BayesOpt


class SimpleBOTest(unittest.TestCase):
    """Simple Bayes. Opt. test case."""

    def test_simple_objective(self):
        """Runs the optimizer for a toy obj. function."""
        x = CSH.UniformFloatHyperparameter(
            "x", lower=0.0, upper=1.0, log=False
        )

        # Define parameters of acq. function, minimization and surrogate.
        acquisition = {"acquisition": "expected_improvement", "xi": 0.01}
        minimizer_kwargs = {"method": "L-BFGS-B", "nrestarts": 25}
        kernel = Matern(
            length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5
        )
        surrogate = {"n_restarts_optimizer": 1, "kernel": kernel}

        # Define simple test obj. function
        def objective(x, noise=0.1):
            noise = np.random.normal(loc=0, scale=noise)
            return (x**3 * np.cos(5 * np.pi * x)) + noise

        bayesopt = BayesOpt(
            space=x,
            objective=objective,
            acquisition=acquisition,
            minimizer_kwargs=minimizer_kwargs,
            surrogate=surrogate,
            log_freq=1,
        )

        # Run Bayes. Opt. for 10 iterations.
        bayesopt.optimize(10)
