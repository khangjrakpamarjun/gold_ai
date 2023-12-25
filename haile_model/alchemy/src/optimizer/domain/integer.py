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
Real dimension module.
"""

from typing import Tuple

import numpy as np

from optimizer.domain.base import BaseDimension


class IntegerDimension(BaseDimension):
    """
    Class for representing a integer valued dimension.

    A simplified version of the skopt API here:
        https://github.com/scikit-optimize/scikit-optimize/blob/master/skopt/space/space.py
    """

    def __init__(self, low: int, high: int):
        """Constructor.

        Args:
            low: int, lower bound (inclusive).
            high: int, upper bound (inclusive).
        """
        if high <= low:
            raise ValueError(f"Lower bound {low} must be less than upper bound {high}.")

        if (low, high) != (int(low), int(high)):
            raise ValueError(f"Both bounds must be integers, got ({low}, {high})")

        self.low = low
        self.high = high

    def _sample(
        self, n_samples: int, random_state: np.random.RandomState
    ) -> np.ndarray:
        """Draw a random sample from the dimension.

        Args:
            n_samples: int, number of samples returned.
            random_state: np.random.RandomSate, seeded rng to do sampling with.

        Returns:
            np.ndarray of samples.
        """
        return random_state.randint(low=self.low, high=self.high + 1, size=n_samples)

    @property
    def bounds(self) -> Tuple:
        """Get the bounds of the dimension.

        Returns:
            Tuple, lower bound, upper bound.
        """
        return self.low, self.high