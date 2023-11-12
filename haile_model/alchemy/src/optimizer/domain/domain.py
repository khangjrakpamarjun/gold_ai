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
Domain module.
"""

import numbers
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from optimizer.domain.base import BaseDimension
from optimizer.domain.categorical import CategoricalDimension
from optimizer.domain.integer import IntegerDimension
from optimizer.domain.real import RealDimension


def check_dimension(
    dimension: Union[
        Tuple[numbers.Real, numbers.Real],
        Tuple[numbers.Real, numbers.Real, str],
        List[Any],
        BaseDimension,
    ]
) -> BaseDimension:
    """Convert the provided dimension specification to a BaseDimension object.

    Also does error checking on `dimension` to be sure that it is a supported type.

    A simplified version of the skopt API here:
        https://github.com/scikit-optimize/scikit-optimize/blob/master/skopt/space/space.py

    Args:
        dimension: dimension specification. Can be:
            - `(low, high)` tuple for a Real or Integer dimension (based on type).
            - `(numbers.Real, numbers.Real, "categorical")` special case for specifying
            a 2 element categorical with two real valued (including integers) elements.
            - List of objects for a Categorical domain.
            - An object extending BaseDimension (Real, Integer, Categorical).

    Raises:
        ValueError if the dimension provided isn't one of the above specifications.

    Returns:
        BaseDimension object.
    """
    if isinstance(dimension, BaseDimension):
        return dimension

    if not isinstance(dimension, (list, tuple, np.ndarray, pd.Series)):
        raise ValueError(
            f"Provided dimension with type `{type(dimension)}` "
            "must be list, tuple, numpy.ndarray, or pandas.Series"
        )

    if isinstance(dimension, pd.Series):
        dimension = np.array(dimension)

    if len(dimension) == 1:
        return CategoricalDimension(dimension)

    if len(dimension) == 2:
        if any(isinstance(d, (str, bool, np.bool_)) for d in dimension):
            dim = CategoricalDimension(dimension)

        elif all(isinstance(d, numbers.Integral) for d in dimension):
            dim = IntegerDimension(*dimension)

        elif all(isinstance(d, numbers.Real) for d in dimension):
            dim = RealDimension(*dimension)

        else:
            raise ValueError(f"Invalid length 2 dimension specification {dimension}.")

        return dim

    if len(dimension) == 3:
        if dimension[2] == "categorical" and all(
            isinstance(d, numbers.Real) for d in dimension[:2]
        ):
            return CategoricalDimension(dimension[:2])

    if len(dimension) >= 3:
        return CategoricalDimension(dimension)

    raise ValueError(f"Invalid dimension {dimension}.")


class Domain:
    """
    Class for handling mixed domain problems.

    A simplified version of the skopt API here:
        https://github.com/scikit-optimize/scikit-optimize/blob/master/skopt/space/space.py
    """

    def __init__(
        self,
        dimensions: List[
            Union[
                Tuple[numbers.Real, numbers.Real],
                Tuple[numbers.Real, numbers.Real, str],
                List[Any],
                BaseDimension,
            ]
        ],
    ):
        """Constructor.

        Args:
            dimensions: List. Each dimension can be:
                * `(low, high)` tuple for a Real or Integer dimension (based on type).
                * `(low, high, "categorical")` to specify a real valued categorical.
                * List of objects for a Categorical domain.
                * An object extending BaseDimension (Real, Integer, Categorical).
        """
        self.dimensions = [check_dimension(d) for d in dimensions]

    def __len__(self) -> int:
        """Get the number of dimensions.

        Returns:
            int.
        """
        return len(self.dimensions)

    def __getitem__(self, item) -> BaseDimension:
        """Get the dimension at the provided index.

        Args:
            item: int, index.

        Returns:
            BaseDimension.
        """
        return self.dimensions[item]

    def sample(
        self,
        n_samples: int = 1,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> np.ndarray:
        """Randomly sample the domain.

        Args:
            n_samples: optional int, number of samples returned.
            random_state: optional int or np.random.RandomSate, sets the random seed of
                the sample operation.

        Returns:
            np.ndarray of sampled points with dimension (n_samples, len(self))
        """
        rng = check_random_state(random_state)

        columns = []

        for d in self.dimensions:
            columns.append(d.sample(n_samples=n_samples, random_state=rng))

        return np.column_stack(columns)
