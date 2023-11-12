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
Stopper which terminates when population meets constraints.
"""
from copy import deepcopy

import numpy as np

from optimizer.problem.problem import OptimizationProblem
from optimizer.solvers.base import Solver
from optimizer.stoppers.base import BaseStopper
from optimizer.stoppers.utils import top_n_indices


class SatisfiedConstraintsStopper(BaseStopper):
    """
    This class will stop the search if all constraints are met.
    It allows the optimization to stop when the `top_n`
    members of the parameter population satisfy *all* the constraints.

    Args:
        top_n: Number of best solutions to check for constraint violations
        sense: Whether to "minimize" or "maximize" the objective.

    Examples
    >> stopper = SatisfiedConstraintsStopper(sense='maximize')
    >> while not solver.stop() or stopper.stop():
    >>      parameters = solver.ask()
    >>      objective_values = problem(parameters)
    >>      solver.tell(parameters, objective_values)
    >>      stopper.update(solver, problem)
    """

    def __init__(self, sense: str, top_n: int = 1):
        self.sense = sense
        self.top_n = top_n

    def update(
        self, solver: Solver, problem: OptimizationProblem, **kwargs
    ):  # pylint: disable=arguments-differ
        """Stop method.
        Returns True when constraints for the top_n best performing
        parameter sets are all met.

        Args:
            solver: Solver object. Maintains signature with other stoppers
            which take Solver objects and kwargs.
            problem: OptimizationProblem object (which contains
            the necessary constraints)
            **kwargs: unused keyword arguments.

        Returns:
            bool, True if the search should stop.
        """
        if problem.penalties is None or not problem.penalties:
            raise ValueError("The OptimizationProblem has no constraints to verify")

        # Evaluating the problem on solver.parameters will change the
        # internal state of problem.penalties. Save the existing penalties obj.
        original_penalties = deepcopy(problem.penalties)
        # Evaluate the problem at the current parameters
        problem(solver.parameters)
        top_idx = top_n_indices(solver.objective_values, self.sense, self.top_n)
        # Make an array where each parameter violates a given constraint.
        constraint_population = np.array(
            [penalty.constraint.violated[top_idx] for penalty in problem.penalties]
        ).T
        # Stop when no members of the population of interest/parameters are violating
        self._stop = constraint_population.sum() == 0

        # reset the original penalties
        problem.penalties = original_penalties
