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
Holds base class for Solvers.
"""

import abc
from copy import deepcopy
from typing import Any, List, Tuple

import numpy as np
from sklearn.utils import check_random_state

from optimizer.types import Matrix, Vector
from optimizer.utils.validation import check_matrix


class Solver(abc.ABC):
    """
    Abstract base class representing a solver.
    Implements the Ask and Tell pattern described in Collette 2010 Chapter 14.

    Main idea is abstract the objective and penalties from the solver
    in a loop such as::

        while not solver.stop():
            # Ask the solver to generate new solutions (could be newly initialized).
            solutions = solver.ask()

            # Evaluate recently generated solutions.
            updated_objectives = objective(solutions)

            # Possibly apply penalty functions.
            updated_objectives += penalties(solutions)

            # Possibly repair solutions.
            repaired_solutions = repairs(solutions)

            # Tell the solver about the newly updated solutions.
            solver.tell(repaired_solutions, updated_objectives)

        # Output the best solution
        return solver.best()

    """

    def __init__(
        self,
        domain: List[Any],
        sense: str = "minimize",
        seed: int = None,
    ):
        """Constructor.

        Args:
            domain: list of describing the domain along each dimension.
            sense: string, optional sense string for "minimize" or "maximize".
            seed: int, optional random seed.
        """
        self._domain = domain

        assert sense in ["minimize", "maximize"]

        self.seed = seed
        self.rng = check_random_state(seed)
        self.told = False
        self.sense = sense

    @abc.abstractmethod
    def ask(self) -> np.ndarray:
        """Get the current state of the solver's parameters.

        Returns: current parameter values.
        """

    @abc.abstractmethod
    def tell(self, parameters: Matrix, objective_values: Vector):
        """Set the internal state of the solver.

        Args:
            parameters: Matrix of updated parameter values to pass to the solver.
            objective_values: Vector of updated objective values to pass to the solver.
        """
        check_matrix(parameters)

    def _process_told_parameters_and_objectives(
        self, parameters: Matrix, objective_values: Vector
    ) -> (np.ndarray, np.ndarray):
        """Convert the parameters and objectives to numpy arrays.
        Convert objectives to a maximization problem if necessary.

        Args:
            parameters: Matrix of updated parameter values to pass to the solver.
            objective_values: Vector of updated objective values to pass to the solver.

        Returns:
            (np.ndarray, np.ndarray), the processed values.
        """
        parameters, objective_values = np.array(parameters), np.array(objective_values)

        if self.sense == "maximize":
            objective_values *= -1

        return parameters, objective_values

    @abc.abstractmethod
    def stop(self) -> bool:
        """Determine if the solver should terminate or not.

        Returns:
            Bool, True is the solver should terminate.
        """

    @abc.abstractmethod
    def best(self) -> Tuple[np.ndarray, float]:
        """Get the current best solution.

        Returns:
            Tuple, (np.ndarray, float), the best solution and its objective value.
        """

    @property
    def objective_values(self) -> Vector:
        """Get the current objective values.

        Returns:
            Vector of objective values.
        """
        objectives = deepcopy(self._internal_objective_values)

        if self.sense == "maximize":
            objectives *= -1

        return objectives

    @property
    @abc.abstractmethod
    def _internal_objective_values(self) -> Vector:
        """Get the current, not sense-checked objective values.

        Returns:
            Vector of objective values.
        """

    @property
    @abc.abstractmethod
    def parameters(self) -> Matrix:
        """Get the current parameter matrix.

        Returns:
            Matrix of parameters.
        """

    @property
    def domain(self) -> List[Any]:
        """Bounds getter.

        Returns:
            List of Tuples.
        """
        return self._domain

    @domain.setter
    def domain(self, value: List[Any]):
        """
        Prevents setting domain with an interpretable message.

        Args:
            value: List.

        Raises:
            NotImplementedError: this method is purposefully disable to avoid
                complexities involved with changing domain during runtime.
        """
        raise NotImplementedError(
            "Solver domain can not be changed. Please create a new Solver instance."
        )
