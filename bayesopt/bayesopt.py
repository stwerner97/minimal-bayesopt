import sys
import logging
from functools import partial
from typing import Callable, Dict, Tuple

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ConfigSpace.hyperparameters as CSH
from scipy.stats import norm
from scipy.optimize import minimize, Bounds
from sklearn.gaussian_process import GaussianProcessRegressor


class BayesOpt:
    """Bayesian Optimization algorithm.

    Implements a Bayesian Optimization algorithm that maximizes the value of
    an objective function w.r.t. a parameter of the search space.

    Parameters
    ----------
    space : CSH.FloatHyperparameter
        Search space of the parameter to optimize.
    objective : Callable (m,) -> (1)
        The objective function to maximize with Bayesian Optimization. Expects
        a function that maps a 1D array (the parameter) to a scalar objective.
    acquisition : Dict
        Dictionary that specifies the acquisition function to use
        (key: `acquisition`) and other parameters.
    minimizer_kwargs : Dict
        Specifies the configuration of the optimizer used to maximize the
        acquisition function. The parameters are passed to `scipy.optimize`.
    surrogate : Dict
        Specifies the configuration of the Gaussian Process Regressor. Its
        parameters are passed to the constructor of the surrogate.
    log_freq : int
        Specifies the frequency at which `BayesOpt` renders the landscape
        of the surrogate and acquisition function.
    seed : int
        The seed of the pseudo random number generator to use.

    Attributes
    ----------
    surrogate : GaussianProcessRegressor
        Probabilistic surrogate of the objective function's landscape.
    xsamples : np.ndarray
        Samples of the search space where the obj. function was evaluated.
    ysamples : np.ndarray
        Objective value of the search space samples in `xsamples`.
    """

    EPS = 1e-8

    def __init__(
        self,
        space: CSH.FloatHyperparameter,
        objective: Callable,
        acquisition: Dict = None,
        minimizer_kwargs: Dict = None,
        surrogate: Dict = None,
        log_freq: int = 1,
        seed: int = None,
    ):
        np.random.seed(seed)

        self.space = space
        self.objective = objective
        self.log_freq = log_freq
        self.seed = seed

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        if acquisition is None:
            acquisition = {"acquisition": "expected_improvement", "xi": 0.01}

        # Pass static parameters such as `xi` to the acquisition function.
        # The function expects only samples of the search space as an input.
        self.acquisition = getattr(self, acquisition.pop("acquisition"))
        self.acquisition = partial(self.acquisition, **acquisition)

        if surrogate is None:
            surrogate = {}

        if minimizer_kwargs is None:
            minimizer_kwargs = {"method": "L-BFGS-B", "nrestarts": 25}

        self.minimizer_kwargs = minimizer_kwargs

        # Standardizes target values, to mitigate problems posed by
        # ill-conditioned covariance matrices.
        self.surrogate = GaussianProcessRegressor(
            random_state=self.seed, normalize_y=True, **surrogate
        )

        self.xsamples = np.empty((0, 1), dtype=np.float32)
        self.ysamples = np.empty((0, 1), dtype=np.float32)

    def optimize(self, niters: int) -> Tuple[float]:
        """Run the Bayesian Optimization loop.

        Iteratively search the surrogate landscape for a sample of the search
        space maximizing the acquisition function, evaluate its objective
        value and refit the surrogate.

        Parameters
        ----------
        niters : int
            The number of iterations of the Bayesian Optimization loop.
            Overall, the objective value is evaluated `niter` times.

        Returns
        -------
        Tuple (float, float)
            The parameter that maximized the objective function during the
            search and its objective value.
        """
        for _ in range(niters):
            xsample = self.search()
            ysample = self.objective(**{self.space.name: xsample})

            xsample = np.asarray(xsample, dtype=np.float32).reshape(1, -1)
            ysample = np.asarray(ysample, dtype=np.float32).reshape(1, -1)

            self.update(xsample, ysample)

            if self.num_xsamples % self.log_freq == 0:
                logging.info(
                    f"Num Evals. {self.num_xsamples}; Max. Obj. Value:"
                    f" {self.yopt:.5f}; Best Sample.: {self.xopt:.5f}"
                )
                self.render()

        return self.xopt, self.yopt

    def search(
        self, nrestarts: int = 25, minimizer_kwargs: Dict = {}
    ) -> float:
        """Search for a maximizer of the acquisition function.

        Search the surrogate landscape for a point that maximizes the
        acquisition function using `scipy.optimize`.

        Parameters
        ----------
        nrestarts : int
            The number of restarts of the L-BFGS-B optimization algorithm.
        minimizer_kwargs : Dict
            Parameters passed to `scipy.optimize`.

        Returns
        -------
        float
            The sample whose acq. function value was maximum during the search.
        """
        if self.num_xsamples <= 0:
            return self.space.rvs()

        # Define an auxiliary function as the negative acquisition function,
        # since SciPy expects a minimization problem.
        def auxiliary(xsample):
            return -1.0 * self.acquisition(xsamples=xsample.reshape(1, -1))

        bounds = Bounds(self.space.lower, self.space.upper)

        minimizer = None

        # Local optimization algorithms (e.g. L-BFGS-B) are sensitive to their
        # starting conditions. Use `nrestarts` random samples of the search
        # space as starting points to the optimization.
        for _ in range(nrestarts):
            x0 = np.random.uniform(
                self.space.lower, self.space.upper, size=(1,)
            )
            result = minimize(
                auxiliary, x0=x0, bounds=bounds, **minimizer_kwargs
            )

            minimizer = result if minimizer is None else minimizer
            minimizer = result if result.fun < minimizer.fun else minimizer

        return float(minimizer.x)

    def update(self, xsample: np.ndarray, ysample: np.ndarray) -> None:
        """Update the surrogate given a sample and its objective value.

        Refits the surrogate given the a sample and its objective value.

        Parameters
        ----------
        xsample : np.ndarray (m, 1)
            Samples of the search space.
        ysample : np.ndarray (m, 1)
            Objective values of the samples.
        """
        self.xsamples = np.vstack((self.xsamples, xsample))
        self.ysamples = np.vstack((self.ysamples, ysample))

        self.surrogate.fit(self.xsamples, self.ysamples)

    def expected_improvement(
        self, xsamples: np.ndarray, xi: float = 0.01
    ) -> np.ndarray:
        """Compute the Expected Improvement (EI) at the passed sampling points.

        Evaluates the EI at sampling points `xsamples` under the probabilistic
        surrogate model. EI trades-off the predicted probab. of improvement
        and the extent to which the objective value is expected to increase.

        Parameters
        ----------
        xsamples : np.ndarray (m, n)
            Sample points at which the EI acquisition function is computed.
        xi : float
            An exploration parameter. The Bayesian Optimization samples points
            of higher uncertainty (standard deviation) as `xi` increases.

        Returns
        -------
        np.ndarray (m, 1)
            EI scores evaluated at sampling points `xsamples`.
        """
        mean, std = self.surrogate.predict(xsamples, return_std=True)
        mean, std = mean.reshape(-1), std.reshape(-1)

        impr = mean - self.yopt - xi
        Z = impr / (std + self.EPS)
        eximpr = impr * norm.cdf(Z) + std * norm.pdf(Z)

        return eximpr

    def render(self, n_samples: int = 200) -> plt.Figure:
        """Show the surrogate and acquisition function landscape.

        Parameters
        ----------
        n_samples : int
            Number of functions samples from the Gaussian Process Regressor.

        Returns
        -------
        plt.Figure
            Figure that plots the surrogate and acquisition function landscape.
        """
        sns.set_style("whitegrid")
        plt.rcParams.update({"font.size": 14})

        fig, axes = plt.subplots(
            nrows=2,
            figsize=(12, 5),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )

        axes[0].set_xlim(self.space.lower, self.space.upper)
        axes[0].set_ylabel("Obj. Value")
        axes[1].set_xlim(self.space.lower, self.space.upper)
        axes[1].set_xlabel(self.space.name)
        axes[1].set_ylabel("Acq. Func.")

        # Plot functions sampled from the surrogate's posterior distribution.
        xsamples = np.linspace(
            self.space.lower, self.space.upper, 1000
        ).reshape(-1, 1)
        mean, std = self.surrogate.predict(xsamples, return_std=True)
        mean, std = mean.reshape(-1), std.reshape(-1)
        ysamples = self.surrogate.sample_y(xsamples, n_samples=n_samples)
        ysamples = np.squeeze(ysamples)

        for ysample in ysamples.T:
            axes[0].plot(xsamples, ysample, alpha=0.1, color="tab:gray")

        # Plot posterior mean & uncertainty estimates.
        axes[0].plot(xsamples, mean, alpha=0.8, color="black")
        axes[0].fill_between(
            xsamples.reshape(-1),
            mean + std,
            mean - std,
            alpha=0.3,
            color="darkorange",
        )

        # Plot samples & objective values of prior evaluations.
        xsampels = self.xsamples.reshape(-1)
        ysamples = self.ysamples.reshape(-1)
        axes[0].scatter(xsampels, ysamples, s=40, color="black", zorder=2.5)

        # Plot marker for best sample found so far.
        axes[0].scatter(
            self.xopt, self.yopt, s=70, color="maroon", zorder=2.5, marker="X"
        )

        # Plot the acquisition function landscape.
        xsamples = np.linspace(
            self.space.lower, self.space.upper, 1000
        ).reshape(-1, 1)
        eximpr = self.acquisition(xsamples=xsamples)
        xsamples, eximpr = xsamples.reshape(-1), eximpr.reshape(-1)
        axes[1].plot(xsamples, eximpr, color="tab:blue")
        axes[1].fill_between(xsamples, eximpr, alpha=0.4, color="tab:blue")

        sns.despine()
        plt.tight_layout()
        return fig

    @property
    def yopt(self) -> float:
        """float: Objective value of the best search space sample."""
        yopt = np.max(self.ysamples, axis=0)
        return float(yopt)

    @property
    def xopt(self) -> float:
        """float: Parameter of the search space with max. objective value."""
        xopt = self.xsamples[np.argmax(self.ysamples, axis=0)]
        return float(xopt)

    @property
    def num_xsamples(self) -> int:
        """int: Number of evaluated samples."""
        return len(self.xsamples)
