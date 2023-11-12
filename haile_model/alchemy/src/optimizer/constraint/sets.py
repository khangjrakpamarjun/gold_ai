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
Constraint set module.
"""

import abc
from numbers import Real
from typing import List, Union

import numpy as np

from optimizer.types import Vector
from optimizer.utils.validation import check_vector


class ConstraintSet(abc.ABC):
    """
    ConstraintSet abstract base class.
    """

    @abc.abstractmethod
    def nearest(self, column: Vector) -> Vector:
        """Determines the closest element in the set for each element in column.

        Args:
            column: single column from a matrix of parameters.

        Returns:
            Vector, same shape as column, the closest elements in the set.
        """
        check_vector(column)

    def distance(self, column: Vector) -> Vector:
        """Determine the distance of the column to its closest element in the set.

        Args:
            column: single column from a matrix of parameters.

        Returns:
            Vector, same shape as column, the distances to closest elements.
        """
        return np.abs(column - self.nearest(column))

    @abc.abstractmethod
    def __str__(self):
        """To string method.

        Returns:
            str.
        """


class MultiplesOf(ConstraintSet):
    """
    Constraint set describing the a multiple of a particular number.
    """

    def __init__(self, multiple: Real):
        """Constructor.

        Args:
            multiple: multiple to enforce.

        Raises:
            ValueError: if multiple equals zero.
        """
        if multiple == 0:
            raise ValueError("Zero is not allowed as a multiple.")

        self.multiple = multiple

    def nearest(self, column: Vector) -> Vector:
        """Gets the nearest multiple of each element in the column.

        Args:
            column: single column from a matrix of parameters.

        Returns:
            Vector, nearest multiple for each element in the column.
        """
        super(MultiplesOf, self).nearest(column)

        return np.round(column / self.multiple) * self.multiple

    def __str__(self):
        """Representation method.

        Returns:
            str.
        """
        return f"Multiples of {self.multiple}"


class Integers(MultiplesOf):
    """
    Set representing the integers.
    """

    def __init__(self):
        """Constructor."""
        super(Integers, self).__init__(1)

    def __str__(self):
        """String method.

        Returns:
            str.
        """
        return "Integers"


class UserDefinedSet(ConstraintSet):
    """
    Allows any collection to be passed.
    """

    def __init__(self, collection: Union[List, Vector]):
        """Constructor.

        Args:
            collection: collection to enforce membership.

        Raises:
            ValueError: if elements in collection are not Reals.
        """
        invalid = [item for item in collection if not isinstance(item, Real)]
        if invalid:
            raise ValueError(
                f"Invalid arguments provided in user defined set: {invalid}"
            )

        self.collection = np.unique(np.array(collection))

    def nearest(self, column: Vector) -> Vector:
        """Find the nearest element in the given set for each element.

        Args:
            column: a single column from a matrix of parameters.

        Returns:
            Vector, the nearest elements in the given set.
        """
        super(UserDefinedSet, self).nearest(column)

        if hasattr(column, "to_numpy"):
            column = column.to_numpy()

        index_of_nearest = np.argmin(
            np.abs(column[:, np.newaxis] - self.collection), axis=1
        )

        return self.collection[index_of_nearest]

    def __str__(self):
        """Str method.

        Returns:
            str.
        """
        return str(list(self.collection))


class StepSizeSet(MultiplesOf):
    """A set for defining steps from a current value that avoids using a UserDefinedSet.

    Creates a constraint set of the form:
        x in {`center` + i * `step_size` | for any real value i}.

    This is equivalent to `MultiplesOf(step_size)` when `center` = 0.0.
    """

    def __init__(self, step_size: Real, center: Real = 0.0):
        """Constructor.

        Args:
            step_size: Real, step size from center.
            center: Real, center value.
        """
        super().__init__(step_size)
        self.center = center

    def nearest(self, column: Vector) -> Vector:
        """Gets the nearest step of each element in the column.

        Args:
            column: single column from a matrix of parameters.

        Returns:
            Vector, nearest multiple for each element in the column.
        """
        return super().nearest(column - self.center) + self.center

    def __str__(self):
        """String method.

        Returns:
            str.
        """
        return f"Size {self.multiple} steps from center {self.center}"
