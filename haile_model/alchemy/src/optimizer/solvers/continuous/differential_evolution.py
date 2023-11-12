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
Differential evolution code.
"""

from numbers import Real
from typing import List, Tuple, Union

import numpy as np
import six
from scipy.optimize._differentialevolution import (
    DifferentialEvolutionSolver as _DifferentialEvolutionSolver,
)

from optimizer.exceptions import InitializationError, MaxIterationError
from optimizer.solvers.base import Solver
from optimizer.types import Matrix, Vector


class _DifferentialEvolutionWrapper(_DifferentialEvolutionSolver):
    """
    Wrapper for the scipy.optimize DifferentialEvolutionSolver.
    Overrides the __next__ method to support the Ask and Tell pattern.
    See here for more:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
    """

    __init_error_msg = (
        "The population initialization method must be one of "
        "'latinhypercube' or 'random', or an array of shape "
        "(M, N) where N is the number of parameters and M>5"
    )

    def __init__(
        self,
        bounds: List[Tuple[Real, Real]],
        seed: int = None,
        popsize: int = None,
        maxiter: int = 1000,
        mutation: Union[Tuple[float, float], float] = (0.5, 1.0),
        recombination: float = 0.7,
        strategy: str = "best1bin",
        init: Union[str, Matrix] = "latinhypercube",
    ):
        """Constructor.

        Args:
            bounds: list of tuples, upper and lower domain for each dimension.
            seed: int, optional random seed.
            popsize: int, optional desired population size.
            maxiter: int, maximum number of iterations.
            mutation: the mutation intensity constant. Denoted by F in literature.
                Must be in the range [0, 2].
                - Tuple[float, float]: dithering used. In other words, the mutation
                    constant is randomly chosen from this range for each generation.
                - float: constant mutation intensity.
            recombination: the recombination (or crossover) constant. Denoted by CR in
                literature. Must be a probability.
            strategy: string describing the mutation strategy (see Scipy docs).
            init: optional str or Matrix.
                - "latinhypercube" or "random" uses specified sampling method.
                - Matrix: a desired initial population. Must have 5 or more rows and
                    len(domain) columns.

        Raises:
            ValueError: if the value provided for init is not a valid string option
                or a matrix of the proper size.
        """
        self.population = None
        self.population_energies = None
        self.scale = None

        init = "latinhypercube" if init is None else init

        super(_DifferentialEvolutionWrapper, self).__init__(
            bounds=bounds,
            func=np.sum,
            seed=seed,
            maxiter=maxiter,
            mutation=mutation,
            recombination=recombination,
            strategy=strategy,
            init=init,
        )

        #
        # If the user wants to set override the population size
        # we have to redo the initialization step outside of scipy.
        #
        if popsize is not None:
            self.num_population_members = max(5, popsize)

            self.population_shape = (self.num_population_members, self.parameter_count)

            if isinstance(init, six.string_types):
                if init == "latinhypercube":
                    self.init_population_lhs()
                elif init == "random":
                    self.init_population_random()
                else:
                    raise ValueError(self.__init_error_msg)
            else:
                self.init_population_array(init)

            self.constraint_violation = np.zeros((self.num_population_members, 1))
            self.feasible = np.ones(self.num_population_members, bool)

        self.nit = 1  # Number of calls to __next__

    def solve(self):
        """Disabled solve method.

        Raises:
            NotImplementedError: this method is purposefully disabled.
        """
        raise NotImplementedError(
            "Solve disabled to enable the the Ask and Tell pattern."
        )

    @property
    def parameters(self) -> np.ndarray:
        """Population getter. Adjusts the population from 0 to 1 to its original scale.

        Returns:
            Matrix representing the current population of the solver.
        """
        return self._scale_parameters(self.population)

    @property
    def objective_values(self) -> np.ndarray:
        """Population energies (fitnesses) getter.
        Purely for naming consistency outside this class.

        Returns:
            Vector representing the current objective values of the solver.
        """
        return self.population_energies

    def update(self, trial_population, trial_fitnesses):
        """Update the population based on the trial vectors' fitnesses.

        Args:
            trial_population: matrix of parameter values.
            trial_fitnesses: vector of objective values.
        """
        # Return parameters values in [0, 1].
        trial_population = self._unscale_parameters(trial_population)

        if np.all(np.isinf(self.population_energies)):
            # Initial population was just evaluated.
            self.population = trial_population
            self.population_energies = trial_fitnesses

        else:
            # Find where the new solutions are better.
            better = trial_fitnesses < self.population_energies

            # Update the population based on the mask.
            self.population = np.where(
                better[:, np.newaxis], trial_population, self.population
            )

            self.population_energies = np.where(
                better, trial_fitnesses, self.population_energies
            )

        # Internals expect the best individual and fitness
        # to be first in their respective lists.
        self._promote_lowest_energy()

    def __next__(self) -> np.ndarray:
        """Get the next trial population.

        Returns:
            Matrix representing the new trial population.

        Raises:
            InitializationError: if called without evaluating the initial population.
            MaxIterationError: when called after max number of iterations.
        """
        self.nit += 1

        if np.all(np.isinf(self.population_energies)):
            # Trial population being generated without
            # evaluating the initial population.
            raise InitializationError(
                "Attempted to generate a trial population "
                "without evaluating the initial population."
            )

        if self.nit < self.maxiter:
            # Dithering chooses a random mutation scale (on by default).
            if self.dither is not None:
                self.scale = (
                    self.random_number_generator.rand()
                    * (self.dither[1] - self.dither[0])
                    + self.dither[0]
                )

            # Create the new population solutions.
            trial_population = np.array(
                [self._mutate(i) for i in range(self.num_population_members)]
            )

            # Ensure the generated solutions are within the domain [0, 1].
            self._ensure_constraint(trial_population)

            return self._scale_parameters(trial_population)

        if self.nit > self.maxiter:
            raise MaxIterationError(
                f"Differential evolution cannot exceed "
                f"its max iteration ({self.maxiter})."
            )

        return self._scale_parameters(self.population)

    def set_bounds(self, bounds: List[Tuple[Real]]):
        """Set the internal domain and updates internal scale values.

        Args:
            bounds: list of tuples.
        """
        self.limits = np.array(bounds).T

        # Update the name mangled private members from the base class.

        # pylint: disable=attribute-defined-outside-init
        self._DifferentialEvolutionSolver__scale_arg1 = 0.5 * (
            self.limits[0] + self.limits[1]
        )
        self._DifferentialEvolutionSolver__scale_arg2 = np.fabs(
            self.limits[0] - self.limits[1]
        )

        # pylint: enable=attribute-defined-outside-init


class DifferentialEvolutionSolver(Solver):
    # pylint: disable=line-too-long
    """
    DifferentialEvolutionSolver. Separate from the wrapper to avoid multiple inheritance.
    Leans on the above wrapper of the scipy DE implementation.
    See here for more:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
    """

    def __init__(
        self,
        domain: List[Tuple[Real, Real]],
        sense: str = "minimize",
        seed: int = None,
        popsize: int = None,
        maxiter: int = 1000,
        mutation: Union[Tuple[float, float], float] = (0.5, 1.0),
        recombination: float = 0.7,
        strategy: str = "best1bin",
        init: Union[str, Matrix] = "latinhypercube",
    ):
        """
        Constructor.

        Args:
            domain: list of tuples, upper and lower domain for each dimension.
            sense: str, "minimize" or "maximize", how to optimize the function.
            seed: int, optional random seed.
        """
        self.solver = _DifferentialEvolutionWrapper(
            bounds=domain,
            seed=seed,
            popsize=popsize,
            maxiter=maxiter,
            mutation=mutation,
            recombination=recombination,
            strategy=strategy,
            init=init,
        )

        super(DifferentialEvolutionSolver, self).__init__(
            domain, sense=sense, seed=seed
        )

    # pylint: enable=line-too-long

    def ask(self) -> np.ndarray:
        """Get the current parameters of population.

        Returns:
            np.array of current population values.
        """
        if not self.told:
            # We have not been told objective values yet.
            # Return the solver's initial population.
            return self.solver.parameters

        # Return the new candidate population for evaluation.
        return next(self.solver)

    def tell(self, parameters: Matrix, objective_values: Vector):
        """Set the population and objective values.
        Updates the internal population based on which
        individuals are performing better.

        Args:
            parameters: Matrix of parameter values representing the population.
            objective_values: Vector of objective values.
        """
        super(DifferentialEvolutionSolver, self).tell(parameters, objective_values)

        parameters, objective_values = self._process_told_parameters_and_objectives(
            parameters, objective_values
        )

        self.solver.update(parameters, objective_values)

        self.told = True  # We have been told the objective values.

    def stop(self) -> bool:
        """Determine if we should stop iterating the Ask and Tell loop.

        Returns:
            Boolean, True if we should stop.
        """
        return not np.all(np.isinf(self.objective_values)) and (
            self.solver.nit >= self.solver.maxiter
        )

    def best(self) -> Tuple[np.ndarray, float]:
        """Get the best solution and its objective value.
        Internal sorting takes care of moving the best solution to the 0 index.

        Returns:
            Vector and float, the solution vector and its objective value.
        """
        return self.parameters[0], self.objective_values[0]

    @property
    def _internal_objective_values(self) -> Vector:
        """Get the internal objective values.

        Returns:
            Vector of the current objective values.
        """
        return self.solver.objective_values

    @property
    def parameters(self):
        """Get the current parameter values.

        Returns:
            Matrix of current parameters.
        """
        return self.solver.parameters
