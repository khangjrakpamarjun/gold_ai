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
Random Search Optimization code.
"""

from copy import deepcopy
from numbers import Real
from typing import List, Tuple, Union

import numpy as np
from six import string_types

from optimizer.domain import RealDimension
from optimizer.exceptions import InitializationError, MaxIterationError
from optimizer.solvers.continuous.base import ScaledContinuousSolver
from optimizer.types import Matrix, Vector
from optimizer.utils.initializer import latin_hypercube, uniform_random


class RandomSearchSolver(ScaledContinuousSolver):
    """Random Search (RS) solver.

    The algorithm works as follows:

    1. Initialize the population with a set of solutions (depends on the init param to
       to the constructor)
    2. For each solution s, sample a neighboring solution s' from a uniform hypercube
       centered on s. The size of this hypercube is 2*epsilon (eps) in each dimension
    3. Evaluate the objective value on s' (user responsibility)
    4. If s' is better than s, then update s.
    5. Repeat from 2 to 4 iteratively.

    As the iterations go on, epsilon is decreased. This is to explicitly address the
    exploration-exploitation trade-off: at the beginning the hypercube is large, and
    thus we explore more broadly the search space. In late iterations epsilon is
    smaller, this forcing us to search intensively in a neighborhood of a (presumably)
    already good-enough solution set.

    The decrease of epsilon is exponential, meaning that in each iteration we update it
    with the rule: eps = (1-decay rate) * eps. There is a possibility to assign a
    minimum threshold below which epsilon is not allowed to decrease.

    If the decay is very high, then we might lose the opportunity to arrive near a
    global optima.
    If the decay is too low, then we'll need more iterations to actually converge to a
    optima.

    The choosing of the epsilon_decay_rate and the min_epsilon is problem specific.
    """

    def __init__(
        self,
        domain: List[Union[Tuple[Real, Real], RealDimension]],
        sense: str = "minimize",
        seed: int = None,
        maxiter: int = 1000,
        popsize: int = None,
        init: Union[str, Matrix] = "random",
        epsilon_decay_rate: float = 0.005,
        min_epsilon: float = 0,
    ):
        """
        Constructor.

        Args:
            domain: list of tuples describing the boundaries of the problem.
            seed: optional random seed for the solver.
            maxiter: maximum number of updates.
            popsize: optional population size.
                If not specified, will be 15 * dimension.
            init: optional initial population.
            epsilon_decay_rate: epsilon is multiplied by (1-rate) in each ask call.
            min_epsilon: the minimum epsilon that once reached stops the exponential
                decay

        Raises:
            InitializationError: if popsize and the row count for the provided initial
                population are not equal.
            ValueError: if the provided initialization function is invalid.
        """
        super(RandomSearchSolver, self).__init__(domain=domain, sense=sense, seed=seed)

        if popsize is None and isinstance(init, string_types):
            # No reason for 15 here other than it being the default
            popsize = 15 * int(np.sqrt(len(domain)))

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
        self.niter = 0

        if isinstance(init, string_types):
            if init == "latinhypercube":
                self._solutions = latin_hypercube((popsize, len(domain)), self.rng)
            elif init == "random":
                self._solutions = uniform_random((popsize, len(domain)), self.rng)
            else:
                raise ValueError(f"Initialization function {init} not implemented.")
        else:
            self._solutions = self._init_population_array(init)

        self.pop_shape = self._solutions.shape

        self._best_solutions = deepcopy(self._solutions)

        self._global_best_idx = None
        self._objective_values = None
        self._best_objective_values = None

        self._epsilon = 1
        self._epsilon_decay_rate = epsilon_decay_rate
        self._min_epsilon = min_epsilon

    def _sample_from_epsilon_hypercube(self) -> Matrix:
        """Take a single random sample from a eps-hypercube centered at 0 of shape self
        ._solutions.shape
        """
        if self._epsilon > self._min_epsilon:
            self._epsilon *= 1 - self._epsilon_decay_rate
        if self._epsilon < self._min_epsilon:
            self._epsilon = self._min_epsilon
        return np.random.uniform(
            low=-self._epsilon, high=self._epsilon, size=self._solutions.shape
        )

    def _update_best_solutions(self):
        """Update the best position for each particle and global best."""
        if self._best_objective_values is None:
            self._best_objective_values = deepcopy(self._objective_values)

        else:
            better = self._objective_values < self._best_objective_values

            self._best_objective_values[better] = self._objective_values[better]
            self._best_solutions[better] = self._solutions[better]

        self._global_best_idx = np.argmin(self._best_objective_values)

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
            return self._inverse_transform_parameters(self._solutions)

        self.niter += 1

        if self.told and self.objective_values is None:
            raise InitializationError(
                "Attempted to update solutions "
                "without evaluating the initial parameters."
            )

        if self.niter > self.maxiter:
            raise MaxIterationError(
                f"Solver cannot exceed its max iteration ({self.maxiter})."
            )

        candidates = self._clip(self._sample_from_epsilon_hypercube() + self._solutions)
        return self._inverse_transform_parameters(candidates)

    def tell(self, parameters: Matrix, objective_values: Vector):
        """Set the population and objective values.
        Updates best histories and global best.

        Args:
            parameters: Matrix of parameter values representing the population.
            objective_values: Vector of objective values.
        """
        super(RandomSearchSolver, self).tell(parameters, objective_values)

        # Convert from Pandas objects to Numpy arrays.
        parameters, objective_values = self._process_told_parameters_and_objectives(
            parameters, objective_values
        )

        self._solutions = self._transform_parameters(parameters)
        self._objective_values = objective_values

        self.told = True  # We have been told the objective values.

        self._update_best_solutions()

        self._solutions = self._best_solutions

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
        return (
            self.parameters[self._global_best_idx],
            self.objective_values[self._global_best_idx],
        )

    @property
    def parameters(self) -> np.ndarray:
        """Parameters getter.

        Returns:
            np.ndarray of current particles.
        """
        return self._inverse_transform_parameters(self._best_solutions)

    @property
    def _internal_objective_values(self) -> np.ndarray:
        """Get internal objective values.

        Returns:
            np.ndarray of current objective values.
        """
        return self._best_objective_values
