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
Ackley function definition.
"""

from typing import List, Tuple

import numpy as np

from .base import TestProblem


class Ackley(TestProblem):
    """Class representing the N-dimensional Ackley function."""

    def __call__(self, parameters) -> np.array:
        """N-dimensional Ackley function.
        https://en.wikipedia.org/wiki/Ackley_function

        Args:
            parameters: np.ndarray, matrix input to the function.

        Returns:
            Vector of objective values.
        """
        n = parameters.shape[-1]
        return (
            -20 * np.exp(-0.2 * np.sqrt(1 / n * np.sum(parameters**2, axis=-1)))
            - np.exp(1 / n * np.sum(np.cos(2 * np.pi * parameters), axis=-1))
            + np.e
            + 20
        )

    @staticmethod
    def bounds(n: int) -> List:
        """Get the bounds for n dimensions.

        Args:
            n: int, dimensions needed.

        Returns:
            List of tuples.
        """
        return [(-32.768, 32.768) for _ in range(n)]

    @staticmethod
    def discrete_domain(n: int) -> List:
        """Get the domain lists for n dimensions. In this discrete domain space the
        best solution is contained.

        Args:
            n: int, dimensions needed.

        Returns:
            List of lists.
        """
        bounds = Ackley.bounds(n)

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
