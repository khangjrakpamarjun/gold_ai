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

"""Genetic algorithm solver definition.
"""

import abc
import numbers
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from six import string_types
from sklearn.utils import check_random_state

from optimizer.domain import (
    CategoricalDimension,
    Domain,
    IntegerDimension,
    RealDimension,
)
from optimizer.domain.base import BaseDimension
from optimizer.exceptions import InitializationError, MaxIterationError
from optimizer.solvers.mixed.base import MixedDomainSolver
from optimizer.types import Matrix, Vector


class CrossoverStrategy(abc.ABC):
    """Base class for crossover strategies."""

    def __init__(
        self,
        crossover_rate: float = 0.8,
        seed: Optional[Union[np.random.RandomState, int]] = None,
    ):
        """Constructor.

        Args:
            crossover_rate: float, probability of performing crossover.
            seed: int or RandomState, optional random seed/state.
        """
        if crossover_rate < 0 or crossover_rate > 1:
            raise ValueError(
                f"Crossover rate must be between 0 and 1, {crossover_rate} given."
            )

        self.crossover_rate = crossover_rate
        self.rng = check_random_state(seed)

    @abc.abstractmethod
    def crossover(
        self,
        parent_population_a: np.ndarray,
        parent_population_b: np.ndarray,
    ) -> np.ndarray:
        """Perform crossover between each parent.

        Args:
            parent_population_a: parent population numpy array.
            parent_population_b: parent population numpy array.

        Returns:
            np.ndarray, the child population.
        """

    @classmethod
    def from_string(cls, strategy: str, **kwargs) -> "CrossoverStrategy":
        """Create a selection strategy from a string.

        Args:
            strategy: str, strategy name.
            **kwargs: keywords to pass to the CrossoverStrategy constructor.

        Returns:
            CrossoverStrategy object.
        """
        if strategy == "one-point":
            return OnePointCrossover(**kwargs)

        else:
            raise ValueError(f"Invalid crossover strategy string {strategy}.")


class OnePointCrossover(CrossoverStrategy):
    """Simplest crossover, picks a single point to cross parents over.

    See here for more:
        https://en.wikipedia.org/wiki/Crossover_%28genetic_algorithm%29
    """

    def crossover(
        self,
        parent_population_a: np.ndarray,
        parent_population_b: np.ndarray,
    ) -> np.ndarray:
        """Perform single point crossover between each parent.

        Args:
            parent_population_a: parent population numpy array.
            parent_population_b: parent population numpy array.

        Returns:
            np.ndarray, the child population.
        """
        children = parent_population_a.copy()

        # Determines which rows we'll perform crossover on.
        crossover_rows = self.rng.rand(children.shape[0]) < self.crossover_rate

        # Generate a matrix with each row equal to [0, 1, ..., children.shape[1] - 1]
        crossover_points = np.arange(children.shape[1]) + np.zeros_like(children)

        # Generate the random points to perform crossover.
        random_points = self.rng.randint(children.shape[1], size=children.shape[0])

        # Create the boolean matrix that corresponds to where we'll incorporate
        # information from `parent_population_b`.
        crossover_mask = np.greater_equal(
            crossover_points, random_points.reshape(-1, 1)
        ) * crossover_rows.reshape(-1, 1)

        children[crossover_mask] = parent_population_b[crossover_mask]
        return children


class SelectionStrategy(abc.ABC):
    """Base class for selection strategies."""

    def __init__(self, seed: Optional[Union[np.random.RandomState, int]] = None):
        """Constructor.

        Args:
            seed: int or RandomState, optional random seed/state.
        """
        self.rng = check_random_state(seed)

    @abc.abstractmethod
    def select(
        self, population: np.ndarray, fitnesses: np.ndarray, n_selected: int = None
    ) -> np.ndarray:
        """Select crossover candidates.

        Args:
            population: ndarray, population to select from.
            fitnesses: ndarray, fitnesses of population.
            n_selected: int, number of candidates to select. Should default to
                2 * population.shape[0].

        Returns:
            ndarray, selected candidates.
        """

    @classmethod
    def from_string(cls, strategy: str, **kwargs) -> "SelectionStrategy":
        """Create a SelectionStrategy object.

        Args:
            strategy: str
            kwargs: keywords to pass to SelectionStrategy constructor.

        Returns:
            SelectionStrategy object.
        """
        if strategy == "tournament":
            return TournamentSelection(**kwargs)

        else:
            raise ValueError(f"Invalid selection strategy string {strategy}.")


class TournamentSelection(SelectionStrategy):
    """Class that implements tournament selection.

    This randomly selects `tournament_size` unique individuals and chooses the best.
    Increasing `tournament_size` will increase "selection pressure".

    See here for more:
        https://en.wikipedia.org/wiki/Tournament_selection
    """

    def __init__(
        self,
        seed: Optional[Union[np.random.RandomState, int]] = None,
        tournament_size: int = 2,
        slip_probability: float = 0.05,
    ):
        """Constructor.

        Args:
            seed: int or RandomState, optional random seed/state.
            tournament_size: int, size of tournament to select with.
            slip_probability: float, probability of _not_ selecting the best individual
                in each tournament. If slip_probability = 0.05, then we'll select the
                best individual in a tournament with 0.95 probability.
        """
        super().__init__(seed=seed)

        if tournament_size < 1:
            raise ValueError(
                f"Tournament size must be positive, {tournament_size} given."
            )

        if slip_probability < 0 or slip_probability > 1:
            raise ValueError(
                f"Slip probability must be between 0 and 1, {slip_probability} given."
            )

        self.tournament_size = tournament_size
        self.slip_probability = slip_probability

    def select(
        self, population: np.ndarray, fitnesses: np.ndarray, n_selected: int = None
    ) -> np.ndarray:
        """Select crossover candidates.

        Returned array is expected to have length 2 * population.shape[0].

        Args:
            population: ndarray, population to select from.
            fitnesses: ndarray, fitnesses of population.
            n_selected: int, number of candidates to select. Should default to
                2 * population.shape[0].

        Returns:
            ndarray, selected candidates.
        """
        if self.tournament_size > population.shape[0]:
            raise ValueError(
                f"Tournament size {self.tournament_size} larger than "
                f"population size {population.shape[0]}. Must be less than or equal."
            )

        out_rows = n_selected or 2 * population.shape[0]

        # Create a out_rows x tournament_size dimensional matrix whose entries are
        # random, unique population indices.
        random_indices = np.argsort(self.rng.rand(out_rows, len(fitnesses)), axis=1)[
            :, : self.tournament_size
        ]

        # Determine the best fitness for each random index.
        best_subset_indices = np.argsort(fitnesses[random_indices], axis=1)

        if self.slip_probability > 0.0:
            # Selects the i-th worst individual with probability (1 - q)q^i.
            # Where q = self.slip_probability.
            column_subset = np.zeros(out_rows, dtype=int)
            for i in range(1, self.tournament_size):
                column_subset[
                    self.rng.rand(out_rows)
                    < (1 - self.slip_probability) * self.slip_probability**i
                ] = i

            best_subset_index = best_subset_indices[np.arange(out_rows), column_subset]

        else:
            # If not slippery, always use the best index.
            best_subset_index = best_subset_indices[:, 0]

        # Convert back to an index in the population.
        best_population_index = random_indices[np.arange(out_rows), best_subset_index]

        return population[best_population_index]


class GeneticAlgorithmSolver(MixedDomainSolver):
    """Extensible genetic algorithm implementation."""

    # pylint: disable=too-many-arguments

    def __init__(
        self,
        domain: Union[
            Domain,
            List[
                Union[
                    Tuple[numbers.Real, numbers.Real],
                    Tuple[numbers.Real, numbers.Real, str],
                    List[Any],
                    BaseDimension,
                ]
            ],
        ],
        sense: str = "minimize",
        seed: int = None,
        maxiter: int = 1000,
        popsize: int = None,
        selection_strategy: Union[str, SelectionStrategy] = "tournament",
        crossover_strategy: Union[str, CrossoverStrategy] = "one-point",
        mutation_rate: float = 0.2,
        selection_kwargs: Dict[str, Any] = None,
        crossover_kwargs: Dict[str, Any] = None,
    ):
        """Constructor.

        Args:
            domain: list of describing the domain along each dimension or Domain object.
            sense: string, optional sense string for "minimize" or "maximize".
            seed: int, optional random seed.
            maxiter: int, total iterations.
            popsize: int, population size.
            selection_strategy: str or SelectionStrategy, how to select solutions for
                crossover and mutation.
            crossover_strategy: str or CrossoverStrategy, how to perform crossover.
            mutation_rate: float [0, 1], the probability any individual will be mutated.
            selection_kwargs: keywords passed to the SelectionStrategy constructor.
            crossover_kwargs: keywords passed to the CrossoverStrategy constructor.
        """
        super().__init__(domain=domain, sense=sense, seed=seed)

        if isinstance(selection_strategy, SelectionStrategy):
            self.selection_strategy = selection_strategy

        elif isinstance(selection_strategy, string_types):
            selection_kwargs = selection_kwargs or {}

            if "seed" not in selection_kwargs:
                selection_kwargs["seed"] = self.rng

            self.selection_strategy = SelectionStrategy.from_string(
                selection_strategy, **selection_kwargs
            )

        else:
            raise ValueError(
                f"Invalid argument for selection_strategy {selection_strategy} with "
                f"type {type(selection_strategy)}. "
                "Must be valid string or SelectionStrategy object."
            )

        if isinstance(crossover_strategy, CrossoverStrategy):
            self.crossover_strategy = crossover_strategy

        elif isinstance(crossover_strategy, string_types):
            crossover_kwargs = crossover_kwargs or {}

            if "seed" not in crossover_kwargs:
                crossover_kwargs["seed"] = self.rng

            self.crossover_strategy = CrossoverStrategy.from_string(
                crossover_strategy, **crossover_kwargs
            )

        else:
            raise ValueError(
                f"Invalid argument for crossover_strategy {crossover_strategy} with "
                f"type {type(crossover_strategy)}. "
                "Must be valid string or CrossoverStrategy object."
            )

        if popsize is None:
            popsize = 15 * int(np.sqrt(len(domain)))

        self.maxiter = maxiter
        self.niter = 1
        self.popsize = popsize
        self.mutation_rate = mutation_rate

        self.best_solution = None
        self.best_objective = float("inf")

        self.domain_object = (
            self.domain if isinstance(self.domain, Domain) else Domain(self.domain)
        )

        self.population = self.domain_object.sample(self.popsize, self.rng)
        self._objective_values = None

        # pylint: enable=too-many-arguments

    def mutate(self, population: np.ndarray) -> np.ndarray:
        """Mutate given population depending on its type.

        Args:
            population: np.ndarray, population to mutate.

        Returns:
            np.ndarray, mutated population.
        """
        # This assumes a small mutation rate.
        # TODO: implement a smarter version when rate is high as this will be slow.
        # TODO: profile this function to understand when a "smarter version" is faster.

        mutate_mask = self.rng.rand(population.shape[0]) < self.mutation_rate

        mutate_indices = mutate_mask.nonzero()[0]

        if mutate_indices.size > 0:
            mutation_columns = self.rng.randint(
                population.shape[1], size=len(mutate_indices)
            )

            for mutation_index, column in zip(mutate_indices, mutation_columns):
                dimension = self.domain_object[column]
                to_mutate = population[mutation_index, column]

                if isinstance(
                    dimension, (RealDimension, IntegerDimension, CategoricalDimension)
                ):
                    if len(dimension.bounds) > 1:
                        mutated = dimension.sample(1, random_state=self.rng).item()

                        while to_mutate == mutated:
                            mutated = dimension.sample(1, random_state=self.rng).item()

                        to_mutate = mutated
                else:
                    raise ValueError(
                        f"Encountered dimension with invalid type {type(dimension)}."
                    )

                population[mutation_index, column] = to_mutate

            return population

        else:
            return population

    def ask(self) -> np.ndarray:
        """Get the current parameters of population.

        Returns:
            np.array of current population values.

        Raises:
            MaxIterationError: when called after maximum iterations.
        """
        if not self.told:
            # First iteration, return the randomly initialized population.
            return self.population

        self.niter += 1

        if self.told and self.objective_values is None:
            raise InitializationError(
                "Attempted to generate children without "
                "evaluating the initial population."
            )

        if self.niter > self.maxiter:
            raise MaxIterationError(
                f"Genetic algorithm cannot exceed its max iteration ({self.maxiter})."
            )

        selected = self.selection_strategy.select(
            self.population, self._objective_values
        )

        children = self.crossover_strategy.crossover(
            selected[: self.popsize], selected[self.popsize :]
        )

        children = self.mutate(children)

        # Always keep the best solution.
        children[0] = self.best_solution

        return children

    def tell(self, parameters: Matrix, objective_values: Vector):
        """Set the population and objective values.

        Args:
            parameters: Matrix of parameter values representing the population.
            objective_values: Vector of objective values.
        """
        super().tell(parameters, objective_values)

        parameters, objective_values = self._process_told_parameters_and_objectives(
            parameters, objective_values
        )

        self.told = True

        # Update bests.
        best_idx = np.argmin(objective_values)
        if objective_values[best_idx] < self.best_objective:
            self.best_objective = objective_values[best_idx]
            self.best_solution = parameters[best_idx]

        self.population = parameters
        self._objective_values = objective_values

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
            self.best_solution,
            self.best_objective if self.sense == "minimize" else -self.best_objective,
        )

    @property
    def _internal_objective_values(self) -> Vector:
        """Get internal objective values.

        Returns:
            np.ndarray of current objective values.
        """
        return self._objective_values

    @property
    def parameters(self) -> Matrix:
        """Parameters getter.

        Returns:
            np.ndarray of current particles.
        """
        return self.population
