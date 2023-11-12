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
Test problem base class.
"""

import abc
from typing import List, Tuple

import numpy as np


class TestProblem(abc.ABC):
    """
    Abstract test problem.
    """

    @abc.abstractmethod
    def __call__(self, parameters) -> np.array:
        """Evaluate the test problem at a list of points.

        Args:
            parameters: matrix of parameters. The solutions to be evaluated.

        Returns:
            Column/row vector of objective values.
        """

    @staticmethod
    @abc.abstractmethod
    def bounds(n: int) -> List[Tuple]:
        """Get the bounds of the problem.

        Args:
            n: int, dimension of problem.

        Returns:
            List of tuples.
        """

    @staticmethod
    @abc.abstractmethod
    def discrete_domain(n: int) -> List[List]:
        """Get the discrete domain for the problem.

        Args:
            n: int, dimension of problem.

        Returns:
            List of lists. Each list is the discrete domain for each variable.
        """

    @staticmethod
    @abc.abstractmethod
    def best(n: int) -> Tuple[np.array, float]:
        """Get the global optimum of the test problem.

        Args:
            n: int, dimension of problem.

        Returns:
            (np.array, float), best solution and its objective value.
        """


class Negated:
    """
    Represents the negated version of a problem.
    """

    def __init__(self, problem: TestProblem):
        """Constructor.

        Args:
            problem: TestProblem to wrap.
        """
        self.problem = problem

    def __call__(self, parameters) -> np.array:
        """Evaluate the test problem at a list of points.

        Args:
            parameters: matrix of parameters. The solutions to be evaluated.

        Returns:
            Column/row vector of objective values.
        """
        return -1 * self.problem(parameters)

    def bounds(self, n: int) -> List[Tuple]:
        """Get the bounds of the problem.

        Args:
            n: int, dimension of problem.

        Returns:
            List of tuples.
        """
        return self.problem.bounds(n)

    def best(self, n: int) -> Tuple[np.array, float]:
        """Get the global optimum of the test problem.

        Args:
            n: int, dimension of problem.

        Returns:
            (np.array, float), best solution and its objective value.
        """
        best, objective_value = self.problem.best(n)
        return best, -1 * objective_value
