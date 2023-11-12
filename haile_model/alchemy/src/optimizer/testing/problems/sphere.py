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
Sphere function definition.
"""

from typing import List, Tuple

import numpy as np

from .base import TestProblem


class Sphere(TestProblem):
    """Class representing the N-dimensional Sphere function."""

    def __call__(self, parameters) -> np.array:
        """N-dimensional Sphere function.

        Args:
            parameters: np.ndarray, matrix input to the function.

        Returns:
            Vector of objective values.
        """
        return np.sum(parameters**2, axis=-1)

    @staticmethod
    def bounds(n: int) -> List[Tuple]:
        """Get the bounds for n dimensions.

        Args:
            n: int, dimensions needed.

        Returns:
            List of tuples.
        """
        return [(-100, 100) for _ in range(n)]

    @staticmethod
    def discrete_domain(n: int) -> List:
        """Get the domain lists for n dimensions. In this discrete domain space the
        best solution is contained.

        Args:
            n: int, dimensions needed.

        Returns:
            List of lists.
        """
        bounds = Sphere.bounds(n)

        return [
            list(np.linspace(bounds[i][0], bounds[i][1], 2 + 2 * i + 1))
            for i in range(n)
        ]

    @staticmethod
    def best(n: int) -> Tuple[np.array, float]:
        """Get the global optimum of the test problem.

        Args:
            n: int, dimension of problem.

        Returns:
            (np.array, float), best solution and its objective value.
        """
        return np.zeros(n), 0.0
