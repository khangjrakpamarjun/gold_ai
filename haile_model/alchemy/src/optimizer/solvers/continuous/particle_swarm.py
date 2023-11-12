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
Particle Swarm Optimization code.
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


class ParticleSwarmSolver(ScaledContinuousSolver):
    """
    Particle Swarm Optimization (PSO) solver.
    Implements the original version of PSO in Kennedy and Eberhart (1995)
    with an inertia term in the velocity update.
    See here for more:
    https://en.wikipedia.org/wiki/Particle_swarm_optimization
    """

    def __init__(
        self,
        domain: List[Union[Tuple[Real, Real], RealDimension]],
        sense: str = "minimize",
        seed: int = None,
        maxiter: int = 1000,
        popsize: int = None,
        init: Union[str, Matrix] = "latinhypercube",
        inertia: float = 0.5,
        social_parameter: float = 2.0,
        cognitive_parameter: float = 2.0,
    ):

        """
        Constructor.
        Inspired in part by:
        scipy.optimize._differentialevolution.DifferentialEvolutionSolver

        Args:
            domain: list of tuples describing the boundaries of the problem.
            seed: optional random seed for the solver.
            maxiter: maximum number of updates.
            popsize: optional population size.
                     If not specified, will be 15 * dimension.
            init: optional initial population.
            inertia: float, multiplier for previous velocity.
            social_parameter: float, multiplier for information taken from population.
                              E.g., the global best in the velocity update equation.
            cognitive_parameter: float, multiplier for past information

        Raises:
            InitializationError: if popsize and the row count for the provided initial
                population are not equal.
            ValueError: if the provided initialization function is invalid.
        """
        super(ParticleSwarmSolver, self).__init__(domain=domain, sense=sense, seed=seed)

        if popsize is None and isinstance(init, string_types):
            # No reason for 15 here other than it being the default
            # for the internals of the DifferentialEvolutionSolver.
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

        self.pop_shape = self._particles.shape

        # Initialize velocities in [-1, 1)^(n x d).
        self._velocities = 2 * uniform_random(self.pop_shape, self.rng) - 1
        self._best_positions = deepcopy(self._particles)

        self.global_best_idx = None
        self._objective_values = None
        self.best_objective_values = None

        self.inertia = inertia
        self.social = social_parameter
        self.cognitive = cognitive_parameter

    def update_velocities(self):
        """Update velocities using global best and interia term.

        Raises:
            InitializationError: if the initial population was not evaluated.
        """
        if not self.told:
            raise InitializationError(
                "Attempted to generate update velocities "
                "without evaluating the initial population."
            )

        global_best = self._particles[self.global_best_idx]

        cognitive_random = self.rng.rand(self.pop_shape[0])[:, np.newaxis]
        social_random = self.rng.rand(self.pop_shape[0])[:, np.newaxis]

        self._velocities = (
            self.inertia * self._velocities
            + self.cognitive
            * cognitive_random
            * (self._best_positions - self._particles)
            + self.social * social_random * (global_best - self._particles)
        )

    def update_best_positions(self):
        """Update the best position for each particle and global best."""
        if self.best_objective_values is None:
            self.best_objective_values = deepcopy(self._objective_values)

        else:
            better = self._objective_values < self.best_objective_values

            self.best_objective_values[better] = self._objective_values[better]
            self._best_positions[better] = self._particles[better]

        self.global_best_idx = np.argmin(self._objective_values)

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
            return self._inverse_transform_parameters(self._particles)

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
        updated = self._clip(self._particles + self._velocities)

        return self._inverse_transform_parameters(updated)

    def tell(self, parameters: Matrix, objective_values: Vector):
        """Set the population and objective values.
        Updates best histories and global best.

        Args:
            parameters: Matrix of parameter values representing the population.
            objective_values: Vector of objective values.
        """
        super(ParticleSwarmSolver, self).tell(parameters, objective_values)

        # Convert from Pandas objects to Numpy arrays.
        parameters, objective_values = self._process_told_parameters_and_objectives(
            parameters, objective_values
        )

        self._particles = self._transform_parameters(parameters)
        self._objective_values = objective_values

        self.told = True  # We have been told the objective values.

        self.update_best_positions()
        self.update_velocities()

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
            self.parameters[self.global_best_idx],
            self.objective_values[self.global_best_idx],
        )

    @property
    def parameters(self) -> np.ndarray:
        """Parameters getter.

        Returns:
            np.ndarray of current particles.
        """
        return self._inverse_transform_parameters(self._particles)

    @property
    def _internal_objective_values(self) -> np.ndarray:
        """Get internal objective values.

        Returns:
            np.ndarray of current objective values.
        """
        return self._objective_values
