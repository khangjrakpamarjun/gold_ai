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
Simulated annealing code.

See this publication for more information:
    https://people.sc.fsu.edu/~inavon/5420a/corana.pdf
"""

from copy import deepcopy
from numbers import Real
from typing import List, Tuple, Union

import numpy as np

from optimizer.domain import RealDimension
from optimizer.exceptions import InitializationError, MaxIterationError
from optimizer.solvers.continuous.base import ScaledContinuousSolver
from optimizer.types import Matrix, Vector
from optimizer.utils.initializer import uniform_random


class SimulatedAnnealingSolver(ScaledContinuousSolver):
    """
    Simulated Annealing.

    This algorithm operates on a single point at once and returns single rowed matrices
    to be compatible with what is expected from other solvers.

    See this publication for more information:
        https://people.sc.fsu.edu/~inavon/5420a/corana.pdf

    This code makes regular references to steps in the algorithm definition above.
    Refer back as needed if the inline comments are not enough information.

    The algorithm detailed above has differences from the usual, combinatorial
    Simulated Annealing algorithm. However, the main ideas remain the same.
    """

    # pylint: disable=too-many-instance-attributes,too-many-arguments

    def __init__(
        self,
        domain: List[Union[Tuple[Real, Real], RealDimension]],
        sense: str = "minimize",
        seed: int = None,
        initial_x: Vector = None,
        initial_temp: float = 1000,
        final_temp: float = 0.1,
        initial_neighborhood: float = 0.1,
        update_scale_steps: int = 20,
        reduce_temp_steps: int = 2,
        maxiter: int = 100000,
        max_acceptable_deterioration: float = 1e-4,
    ):
        """Constructor.

        Args:
            domain: list of tuples, upper and lower domain for each dimension.
            sense: string, optional sense string for "minimize" or "maximize".
            seed: int, optional random seed.
            initial_x: optional vector for the initial point.
                If this is not provided, a random point in the domain will be chosen.
            initial_temp: optional initial temperature. This controls how likely it is a
                worse performing solution will be accepted. This is usually large.
            final_temp: optional final temperature. The temperature will be reduced on
                an exponential decay each reduction step.
            initial_neighborhood: ratio of domain to use as initial neighborhood.
            update_scale_steps: (N_s) number of random moves before adjusting the step
                vector.
            reduce_temp_steps: (N_T) number of step vector adjustments before reducing
                the temperature and setting the current point to the best found so far.
            maxiter: maximum number of iterations.
            max_acceptable_deterioration: (epsilon) the max difference from the current
                best objective at the temperature reduction that will allow the current
                solution to continue as the point to generate solutions with.
        """
        super(SimulatedAnnealingSolver, self).__init__(domain, sense=sense, seed=seed)

        n = len(domain)

        if initial_x is not None:
            initial_x = np.array(initial_x).squeeze()

            if initial_x.ndim != 1 or len(initial_x) != len(domain):
                raise ValueError(
                    "Provided initial point must be a 1-dimensional Vector with the "
                    "same number of entries as domain."
                )

            initial_x = self._transform_parameters(initial_x)

            if not ScaledContinuousSolver.in_zero_one(initial_x):
                raise InitializationError(
                    "Provided initial point must be within the domain of the problem."
                )

            self._initial_x = initial_x

        else:
            self._initial_x = uniform_random((n,), self.rng)

        self.best_point = None
        self.best_objective = np.PINF

        self.current_point = None
        self.current_objective = np.PINF

        self.temperature = initial_temp

        if initial_neighborhood < 0 or initial_neighborhood > 1:
            raise ValueError(
                f"The initial neighborhood must be in [0, 1], "
                f"{initial_neighborhood} given."
            )

        self.initial_neighborhood = initial_neighborhood
        self.step_vector = np.full(n, self.initial_neighborhood)

        self.update_scale_steps = update_scale_steps
        self.reduce_temp_steps = reduce_temp_steps

        if initial_temp <= 0 or final_temp <= 0:
            raise ValueError(
                "Final and initial temperatures must be greater than zero."
            )

        if initial_temp <= final_temp:
            raise ValueError(
                "The initial temperature must be greater than the final temperature."
            )

        # Calculate the reduction coefficient so it will reach final_temp at maxiter.
        self.initial_temp = initial_temp
        self.final_temp = final_temp

        number_temp_reductions = maxiter // (
            self.reduce_temp_steps * self.update_scale_steps * n
        )

        if number_temp_reductions == 0:
            raise ValueError(
                "Number of temperature reductions is zero, "
                "increase maxiter, decrease reduce_temp_steps, "
                "or decrease update_scale_steps."
            )

        self.reduction_coefficient = (self.final_temp / self.initial_temp) ** (
            1 / number_temp_reductions
        )

        self.current_dimension = 0
        # Number of steps along a particular dimension that produced a better solution.
        self.successful_steps = np.zeros(n)
        self.n_cycles = 0  # Number times we have taken a step in every direction.
        self.n_step_adjustments = 0
        self.maxiter = maxiter
        self.niters = 1
        self.max_acceptable_deterioration = max_acceptable_deterioration
        self.max_generation_tries = 1000

    # pylint: enable=too-many-instance-attributes,too-many-arguments

    def _generate_point(self) -> Vector:
        """Generate a point by varying the current point along one dimension.

        Returns:
            Vector.
        """
        point = deepcopy(self.current_point)

        r = self.rng.random_sample() * 2 - 1  # r in [-1, 1).
        point[self.current_dimension] += r * self.step_vector[self.current_dimension]
        return point

    def ask(self) -> np.ndarray:
        """Get the current point.

        Returns:
            Matrix with a single row of parameters.

        Raises:
            RuntimeError: when a point cannot be generated in the given domain.
        """
        if self.niters > self.maxiter:
            raise MaxIterationError(
                f"Simulated annealing cannot exceed its max iteration ({self.maxiter})."
            )

        if self.told and self.current_point is None:
            raise InitializationError(
                "Attempted to generate a new population without evaluating first."
            )

        if not self.told:
            return self.initial_x[np.newaxis, :]

        # Steps 1 and 2.
        # Generate a new point until it is in the domain.
        for _ in range(self.max_generation_tries):
            point = self._generate_point()

            if self.in_zero_one(point):
                return self._inverse_transform_parameters(point)[np.newaxis, :]

        raise RuntimeError(
            "Simulated annealing could not generate a point within the "
            "given domain. Try reducing step_vector_initial_ratio or widening "
            "the domain of the problem."
        )

    def tell(  # pylint: disable=too-many-statements
        self, parameters: Matrix, objective_values: Vector
    ):
        """Update the next point and internal state.

        Args:
            parameters: Matrix with a single row of parameter values.
            objective_values: Vector with a single objective value.
        """
        super(SimulatedAnnealingSolver, self).tell(parameters, objective_values)

        parameters, objective_values = self._process_told_parameters_and_objectives(
            parameters, objective_values
        )

        if parameters.shape != (1, len(self.domain)):
            raise ValueError(
                f"Got parameters shape {parameters.shape}, "
                f"required (1, {len(self.domain)})."
            )

        parameters = self._transform_parameters(parameters).squeeze()
        objective_value = objective_values.item()

        # Step 3.
        # Accept the new point if it improves on our current point...
        if objective_value <= self.current_objective:
            self.current_point = deepcopy(parameters)
            self.current_objective = deepcopy(objective_value)
            self.successful_steps[self.current_dimension] += 1

            if objective_value < self.best_objective:
                self.best_point = deepcopy(parameters)
                self.best_objective = deepcopy(objective_value)

        # ... or if the Metropolis criterion is satisfied.
        # Note: we have that current_obj < obj_value, so numerator is already negative.
        elif self.rng.random_sample() < np.exp(
            (self.current_objective - objective_value) / self.temperature
        ):
            self.current_point = deepcopy(parameters)
            self.current_objective = deepcopy(objective_value)
            self.successful_steps[self.current_dimension] += 1

        # Step 4.
        # Increment the current dimension and loop back if necessary.
        self.current_dimension += 1
        if self.current_dimension >= len(self.domain):
            self.current_dimension = 0
            self.n_cycles += 1

        # Step 5.
        # If we have cycled enough, update the step vector based on successful steps.
        if self.n_cycles >= self.update_scale_steps:
            percent_sucessful = self.successful_steps / self.update_scale_steps

            explore = self.step_vector * (1 + 2 * (percent_sucessful - 0.6) / 0.4)
            exploit = self.step_vector / (1 + 2 * (0.4 - percent_sucessful) / 0.4)

            # Explore more along dimensions that steps are more often unsuccessful.
            explore_mask = self.successful_steps > 0.6 * self.update_scale_steps
            self.step_vector[explore_mask] = explore[explore_mask]

            # Exploit more along dimensions that steps are more often successful.
            exploit_mask = self.successful_steps < 0.4 * self.update_scale_steps
            self.step_vector[exploit_mask] = exploit[exploit_mask]

            # Replace values that became too large with the initial value.
            self.step_vector[self.step_vector > 1] = self.initial_neighborhood

            # Update counters.
            self.n_cycles = 0
            self.successful_steps = np.zeros_like(self.step_vector)
            self.n_step_adjustments += 1

        # Step 6.
        # If we have adjusted the step vector enough, reduce the temperature.
        # Occurs every update_scale_steps * reduce_temp_steps cycles.
        if self.n_step_adjustments >= self.reduce_temp_steps:
            self.temperature *= self.reduction_coefficient

            self.n_step_adjustments = 0

            # Step 7 (only occurs if we adjust temperature).
            # Determine if the algorithm should stop or go back to the best found point.
            optimal_change_small = (
                objective_value - self.best_objective
                <= self.max_acceptable_deterioration
            )

            # If our current objective
            if not optimal_change_small:
                self.current_point = deepcopy(self.best_point)
                self.current_objective = deepcopy(self.best_objective)

        self.niters += 1
        self.told = True

        # pylint: enable=too-many-statements

    def stop(self) -> bool:
        """Determine if we should stop iterating.

        Returns:
            Boolean, True if we should stop.
        """
        return self.niters >= self.maxiter

    def best(self) -> Tuple[np.ndarray, float]:
        """Get the best point and objective.

        Returns:
            The best found point and its objective.

        Raises:
            AttributeError: if no point have been evaluated.
        """
        if self.best_point is None:
            raise AttributeError("No points have been evaluated yet.")

        return (
            self._inverse_transform_parameters(self.best_point),
            self.best_objective
            if self.sense == "minimize"
            else -1 * self.best_objective,
        )

    @property
    def _internal_objective_values(self) -> Vector:
        """Return the internal point's objective value.

        Returns:
            The current point's objective value as an array.
        """
        return np.array(self.current_objective)

    @property
    def parameters(self) -> Matrix:
        """Get the current parameter values.

        Returns:
            Matrix (single row) of parameters.
        """
        return self._inverse_transform_parameters(self.current_point)[np.newaxis, :]

    @property
    def initial_x(self) -> Vector:
        """Get the initial point.

        Returns:
            Vector.
        """
        return self._inverse_transform_parameters(self._initial_x).squeeze()
