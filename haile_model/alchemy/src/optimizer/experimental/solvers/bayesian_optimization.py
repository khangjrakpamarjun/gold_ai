# Copyright (c) 2016 - present
# QuantumBlack Visual Analytics Ltd (a McKinsey company).
# All rights reserved.
#
# This software framework contains the confidential and proprietary information
# of QuantumBlack, its affiliates, and its licensors. Your use of these
# materials is governed by the terms of the Agreement between your organisation
# and QuantumBlack, and any unauthorised use is forbidden. Except as otherwise
# stated in the Agreement, this software framework is for your internal use
# only and may only be shared outside your organisation with the prior written
# permission of QuantumBlack.

"""
Bayesian Optimizer Solver code.
"""
import importlib
from numbers import Real
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from six import string_types
from sklearn.gaussian_process import GaussianProcessRegressor

from optimizer.domain import RealDimension
from optimizer.exceptions import InitializationError, MaxIterationError
from optimizer.solvers.continuous.base import ContinuousSolver
from optimizer.types import Matrix, Vector
from optimizer.utils.initializer import latin_hypercube, uniform_random


class BayesianOptSolver(ContinuousSolver):
    """
    Bayesian Optimization Solver.
    """

    def __init__(
        self,
        domain: List[Union[Tuple[Real, Real], RealDimension]],
        sense: str = "minimize",
        seed: int = None,
        maxiter: int = 1000,
        popsize: int = None,
        init: Union[str, Matrix] = "latinhypercube",
        gpr_kwargs: Dict = None,
        acq_func: str = "ucb",
    ):

        """
        Constructor of bayesian optimization solver.


        Args:
            domain: list of tuples describing the boundaries of the problem.
            seed: optional random seed for the solver.
            maxiter: maximum number of updates.
            popsize: optional population size.
                     If not specified, will be 2 * dimension.
            init: optional initial population.
            gpr_kwargs: A dictionary of keyword args to pass to the internal GP
            model that's serves as our surrogate model. As the kernel function can
            be arbitrary, the 'kernel' entry is itself a dictionary, with structure
            like: ::
                * kernel:
                    class: sklearn.gaussian_process.kernels.Matern
                    kwargs:
                        nu: 2.5
                        length_scale: 0.1

            acq_func: Acquisition function. Either ucb or expected improvement.

        Raises:
            InitializationError: if popsize and the row count for the provided initial
                population are not equal.
            ValueError: if the provided initialization function is invalid.
        """
        super(BayesianOptSolver, self).__init__(domain=domain, sense=sense, seed=seed)

        if popsize is None and isinstance(init, string_types):
            # No reason for 15 here other than it being the default
            # for the internals of the DifferentialEvolutionSolver.
            popsize = 2 * int(np.sqrt(len(domain)))

        if (
            popsize is not None
            and not isinstance(init, string_types)
            and popsize != init.shape[0]
        ):
            raise InitializationError(
                f"{type(self).__name__} provided popsize and initial "
                f"population with mismatched sizes. Remove popsize "
                f"keyword argument or set popsize = init.shape[0]."
            )

        self.maxiter = maxiter
        self.niter = 1

        if isinstance(init, string_types):
            if init == "latinhypercube":
                self._particles = latin_hypercube((popsize, len(domain)), self.rng)
            elif init == "random":
                self._particles = uniform_random((popsize, len(domain)), self.rng)
            else:
                raise ValueError(f"Initialization function {init} not implemented.")
        else:
            self._particles = self._init_population_array(init)
        self.gpr_kwargs = gpr_kwargs
        if gpr_kwargs is None:
            self.gpr_kwargs = {}
        # Update gpr_kwargs
        kernel_args_dict = self.gpr_kwargs.get("kernel", {})
        if kernel_args_dict.get("class") is not None:
            kernel_class = load_obj(kernel_args_dict.get("class"))
            self.gpr_kwargs["kernel"] = kernel_class(**kernel_args_dict.get("kwargs"))
        self._gpr = GaussianProcessRegressor(**self.gpr_kwargs)

        if acq_func == "expected_improvement":
            self._acq_func = expected_improvement
        elif acq_func == "ucb":
            self._acq_func = ucb
        else:
            raise ValueError(
                "Only acquisition functions implemented are `ucb` and "
                "`expected_improvement`"
            )
        self.global_best_idx = None
        self._objective_values = None
        self.best_objective_values = None

    def ask(self) -> np.ndarray:
        """Get the current parameters of population.

        Returns:
            np.array of current population values.

        Raises:
            MaxIterationError: when called after maximum iterations.
        """
        if not self.told:
            # We have not been told objective values yet.
            # Return the solver's initial population.
            return self._particles

        self.niter += 1

        if self.told and self.objective_values is None:
            raise InitializationError(
                "Attempted to update particle positions "
                "without evaluating the initial population."
            )

        if self.niter > self.maxiter:
            raise MaxIterationError(
                f"Particle swarm cannot exceed its max iteration ({self.maxiter})."
            )

        # Update positions and return resulting population matrix.
        new_particles = self._acq_func_opt()
        if np.any(np.abs(new_particles - self._particles) <= 1e-7):
            domain_ = np.array(self.domain)
            new_particles = np.random.uniform(
                domain_[:, 0], domain_[:, 1], domain_.shape[0]
            ).reshape(1, -1)
        return new_particles

    def _acq_func_opt(self, n_restarts=25):
        domain = np.array(self.domain)
        dim = domain.shape[0]
        restart_points = np.random.uniform(
            domain[:, 0], domain[:, 1], size=(n_restarts, dim)
        )
        y_best = 1.0

        return_best = np.zeros((dim))
        objectives = self._objective_values
        for pt in restart_points:
            x_opt = minimize(
                self._acq_func,
                x0=pt,
                bounds=domain,
                method="L-BFGS-B",
                args=(self._gpr, self._particles, objectives, self.sense),
            )
            if x_opt.fun < y_best:
                y_best = x_opt.fun[0]
                return_best = x_opt.x
        return return_best.reshape(1, -1)

    def tell(self, parameters: Matrix, objective_values: Vector):
        """Set the population and objective values.
        Update global best and refit internal gp estimator model

        Args:
            parameters: Matrix of parameter values representing the population.
            objective_values: Vector of objective values.
        """
        super(BayesianOptSolver, self).tell(parameters, objective_values)

        # Convert from Pandas objects to Numpy arrays.
        parameters, objective_values = self._process_told_parameters_and_objectives(
            parameters, objective_values
        )
        """
        If i've been told before, update with new positions, otherwise
        particles are the starting population."""
        if not self.told:
            self._particles = parameters
            self._objective_values = objective_values
        else:
            self._particles = np.concatenate((self._particles, parameters), axis=0)
            self._objective_values = np.concatenate(
                (self._objective_values, objective_values), axis=0
            )
        # Re-fit model
        self._gpr.fit(parameters, objective_values)

        self.told = True  # We have been told the objective values.

    def stop(self) -> bool:
        """Determine if the search has reached max iterations.

        Returns:
            True if we should stop searching.
        """
        return self.niter >= self.maxiter

    def best(self) -> Tuple[np.ndarray, float]:
        """Get the best solution found so far.

        Returns:
            Vector and float, the solution vector and its objective value.
        """
        self.global_best_idx = np.argmin(self._objective_values)
        return (
            self.parameters[self.global_best_idx],
            self.objective_values[self.global_best_idx],
        )

    @property
    def parameters(self) -> np.ndarray:
        """Parameters getter.

        Returns:
            np.ndarray of current particles.
        """
        return self._particles

    @property
    def _internal_objective_values(self) -> np.ndarray:
        """Get internal objective values.

        Returns:
            np.ndarray of current objective values.
        """
        return self._objective_values


def ucb(
    x_test: np.ndarray,
    gp_model: Any,
    X_s: np.ndarray,
    Y_s: np.ndarray,
    sense: str,
    beta: float = 9.0,
):  # pylint: disable=unused-argument
    """
    Upper confidence bound acquisition function

    Args:
        x_test: test point
        gp_model: Surrogate model
        X_s: nx1 np.array of evaluated points
        Y_s: nx1 np.array of evaluated points
        sense: maximize or minimize
        beta: exploration parameter

    Returns: upper confidence bound
    """
    x_test = x_test.reshape(1, -1)
    mu, sigma = gp_model.predict(x_test, return_std=True)
    return mu + np.sqrt(beta) * sigma


def expected_improvement(
    x_test: np.ndarray,
    gp_model: Any,
    X_s: np.ndarray,
    Y_s: np.ndarray,
    sense: str,
    psi: float = -1.0,
):  # pylint: disable=unused-argument
    """
    Expected Improvement acquisition: Still needs to be tested

    Args:
        x_test: test point
        gp_model: Surrogate model
        X_s: nx1 np.array of evaluated points
        Y_s: nx1 np.array of evaluated points
        sense: maximize or minimize
        psi: exploration/exploitation parameter

    Returns: Expected improvement at test point
    """
    x_test = x_test.reshape(1, -1)
    f_best = np.min(Y_s)
    mu, sigma = gp_model.predict(x_test, return_std=True)

    ei = (mu - f_best - psi) * norm.cdf(
        f_best - psi, loc=mu, scale=sigma
    ) + sigma * norm.pdf(f_best - psi, loc=mu, scale=sigma)
    # Return negative improvement because we're minimizing
    ei[sigma == 0] = 0
    return -ei


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.

    Args:
        obj_path: Path to an object to be extracted, including the object name.
        default_obj_path: Default object path. In the case this is provided, `obj_path`
        must be a single name of the object being imported.

    Returns:
        Extracted object.

    Raises:
        AttributeError: When the object does not have the given named attribute.

    Examples:
        Importing an object::

            >>> load_obj("sklearn.linear_model.Ridge")

        Importing using `default_obj_path`::

            >>> load_obj("Ridge", default_obj_path="sklearn.linear_model")
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            "Object `{}` cannot be loaded from `{}`.".format(obj_name, obj_path)
        )
    return getattr(module_obj, obj_name)
