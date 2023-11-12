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
Handler module.
"""
import abc
import re
from typing import Callable

from optimizer.constraint.constraint import BaseConstraint
from optimizer.types import Vector


class BaseHandler(abc.ABC):
    """
    Represents a constraint handling class.
    """

    def __init__(self, constraint_: BaseConstraint):
        """Constructor.

        Args:
            constraint_: BaseConstraint object.
        """
        self.constraint = constraint_

    @property
    def constraint_values(self) -> Vector:
        """Get the handler's constraint values from the last evaluation.

        Returns:
            Vector of constraint values.
        """
        return self.constraint.constraint_values

    @property
    def distances(self) -> Vector:
        """Get the handler's constraint distances.

        Returns:
            Vector of distance values.
        """
        return self.constraint.distances

    @property
    def constraint_func(self) -> Callable:
        """Get the handler's constraint function.

        Returns:
            Callable.
        """
        return self.constraint.constraint_func

    @property
    def name(self) -> str:
        """Get the handler's constraint name with the type of handler appended.

        Returns:
            str.
        """
        name_ = self.constraint.name

        return (
            name_
            if name_ is None
            else "_".join(
                [name_]
                + list(
                    map(
                        str.lower,
                        re.findall(r"[A-Z][^A-Z]*", self.__class__.__name__),
                    )
                )
            )
        )
