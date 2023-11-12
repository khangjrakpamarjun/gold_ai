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

"""Continuous solver base classes.
"""
from numbers import Real
from typing import List, Tuple, Union

import numpy as np

from optimizer.domain import RealDimension, check_continuous_domain
from optimizer.solvers import Solver
from optimizer.types import Vector


def to_limits(domain: List[Tuple[Real, Real]]) -> np.array:
    """Convert a list of tuples to a matrix describing the boundaries.

    If the tuples passed in are:
        [(lo_1, hi_1), ... (lo_d, hi_d)]

    The Numpy array output will be of the form:
        [[lo_1, ..., lo_d],
         [hi_1, ..., hi_d]]

    Args:
        domain: list of tuples.

    Returns:
        np.array.
    """
    return np.array(list(zip(*domain)))


class ContinuousSolver(Solver):
    # pylint: disable=abstract-method
    def __init__(
        self,
        domain: List[Union[Tuple[Real, Real], RealDimension]],
        sense: str = "minimize",
        seed: int = None,
    ):
        """Constructor.

        Args:
            domain: list of tuples or RealDimension, upper and lower values or
                RealDimension instance specifying the bounds of each dimension.
            sense: string, optional sense string for "minimize" or "maximize".
            seed: int, optional random seed.
        """
        super(ContinuousSolver, self).__init__(
            check_continuous_domain(domain), sense=sense, seed=seed
        )

    # pylint: enable=abstract-method


class ScaledContinuousSolver(ContinuousSolver):
    # pylint: disable=abstract-method
    """
    Base class for a search algorithm with an internal
    group of solutions that are scaled between [0, 1].

    Inspired by some of the code optimizations/simplifications in:
        scipy.optimize._differentialevolution.DifferentialEvolutionSolver.
    """

    def __init__(
        self,
        domain: List[Union[Tuple[Real, Real], RealDimension]],
        sense: str = "minimize",
        seed: int = None,
    ):
        """Constructor.

        Args:
            domain: list of tuples, upper and lower domain for each dimension.
            sense: string, optional sense string for "minimize" or "maximize".
            seed: int, optional random seed.
        """
        self.limits, self._scale_mean = None, None
        self._scale_range_multiplier, self._scale_range_divisor = None, None

        super(ScaledContinuousSolver, self).__init__(domain, sense=sense, seed=seed)

        self._set_domain_and_scale(domain)

    @staticmethod
    def in_zero_one(point: Vector) -> bool:
        """Determines if a point is in the range [0, 1]^n.

        Args:
            point: Vector of values.

        Returns:
            True if the point is in [0, 1]^n.
        """
        point = point.squeeze()

        return all((point >= 0) & (point <= 1))

    def _init_population_array(self, parameters: np.ndarray) -> np.ndarray:
        """Ensure the provided initial population is within the boundaries.

        Args:
            parameters: user provided initial population array.

        Returns:
            np.ndarray in the range [0, 1].
        """
        parameters = np.array(parameters)

        return self._clip(self._transform_parameters(parameters))

    def _inverse_transform_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """Scale parameters from [0, 1] to original scale.

        Args:
            parameters: np.ndarray in range [0, 1]^(n x d).

        Returns:
            np.ndarray in the original scale of the problem.
        """
        return self._scale_mean + (parameters - 0.5) * self._scale_range_multiplier

    def _transform_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """Scale parameters from original scale to [0, 1].

        Args:
            parameters: np.ndarray in original scale.

        Returns:
            np.ndarray in the range [0, 1]^(n x d).
        """
        return (parameters - self._scale_mean) / self._scale_range_divisor + 0.5

    def _clip(self, parameters: np.ndarray) -> np.ndarray:
        """Clip given parameters to be between [0, 1].

        Args:
            parameters: matrix to clip.

        Returns:
            np.ndarray in [0, 1]^(n x d).
        """
        return np.clip(parameters, 0, 1)

    def _set_domain_and_scale(self, value: List[Tuple[Real, Real]]):
        """Maintains the internal scale values and sets the domain tuples.

        Args:
            value: List of Tuples.
        """
        self._domain = value

        self.limits = to_limits(value)

        # These will be used to scale the population from/to [0, 1].
        self._scale_mean = 0.5 * (self.limits[0] + self.limits[1])

        scale_range = np.fabs(self.limits[0] - self.limits[1])
        self._scale_range_multiplier = scale_range

        # Protect from divide by zero.
        self._scale_range_divisor = np.where(scale_range == 0, 1, scale_range)

    # pylint: enable=abstract-method
