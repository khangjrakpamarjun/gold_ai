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
Repair module.
"""
import abc
from copy import deepcopy
from typing import Callable, Union

import numpy as np

from optimizer.constraint.constraint import (
    EqualityConstraint,
    InequalityConstraint,
    SetMembershipConstraint,
    constraint,
)
from optimizer.constraint.handler import BaseHandler
from optimizer.exceptions import InvalidConstraintError
from optimizer.types import Matrix
from optimizer.utils.functional import safe_assign_column, safe_assign_rows


class Repair(BaseHandler):
    """
    Abstract base class for a general repair.
    """

    @abc.abstractmethod
    def __call__(self, parameters: Matrix) -> Matrix:
        """Repair and return a matrix.

        Args:
            parameters: Matrix to repair.

        Returns:
            The repaired matrix.
        """


class SetRepair(Repair):
    """
    Represents a repair function for a single column.
    """

    def __init__(self, constraint_: SetMembershipConstraint):
        """Constructor

        Args:
            constraint_: SetMembershipConstraint object representing the
                constraint being repaired.

        Raises:
            InvalidConstraintError: if the constraint function is not callable.
        """
        super().__init__(constraint_)

        if not callable(constraint_.constraint_func):
            raise InvalidConstraintError("Given constraint function is not callable.")

    def __call__(self, parameters: Matrix) -> Matrix:
        """Repair and return a matrix.

        Args:
            parameters: Matrix to be repaired.

        Returns:
            The repaired matrix.
        """
        self.constraint(parameters)

        repaired_values = self.constraint.constraint_set.nearest(  # noqa
            self.constraint_values
        )

        return safe_assign_column(
            parameters,
            repaired_values,
            self.constraint.constraint_func.keywords["col"],
            row_mask=self.constraint.violated,
        )


_CHECK_NEVER = "never"
_CHECK_ONCE = "once"
_CHECK_ALWAYS = "always"


class UserDefinedRepair(Repair):
    """
    Represents a repair for a non-set membership constraint.
    """

    def __init__(
        self,
        constraint_: Union[InequalityConstraint, EqualityConstraint],
        repair_function: Callable[[Matrix], Matrix],
        check_repaired: str = _CHECK_ALWAYS,
    ):
        """Constructor.

        Args:
            constraint_: constraint object defining which solutions are infeasible.
            repair_function: function used to repair solutions.
            check_repaired: repair function strictness. Checking methods are
            defined as "never" to never check that the repair function correctly
                "once" to only check that the repair function correctly repaired once on
                the first time its called, and "always" to always check that the  repair
                function correctly repaired the solutions.
        """
        super().__init__(constraint_)

        self.repair_function = repair_function
        self.check_repaired = check_repaired
        self.called = False

    def __call__(self, parameters: Matrix) -> Matrix:
        """Repair and return a matrix.

        Args:
            parameters: Matrix to be repaired.

        Raises:
            RuntimeError:

        Returns:
            The repaired matrix.
        """
        self.constraint(parameters)

        # Only call the repair function if we violated the constraint.
        if any(self.constraint.violated):
            repaired = safe_assign_rows(
                parameters,
                self.repair_function(deepcopy(parameters)),
                self.constraint.violated,
            )

        else:
            repaired = parameters

        if self.check_repaired == _CHECK_ALWAYS or (
            self.check_repaired == _CHECK_ONCE and not self.called
        ):
            distances = self.constraint.calculate_distances(
                self.constraint.constraint_func(repaired), repaired
            )

            # Allow for rounding error in greater than 0 comparison.
            if any(distances > np.sqrt(np.finfo(float).eps)):
                name_str = (
                    "constraint " + self.constraint.name
                    if self.constraint.name is not None
                    else "unnamed constraint"
                )

                raise RuntimeError(
                    f"Repaired parameters still violate {name_str}. This error "
                    f'is thrown when using check_repaired="{self.check_repaired}". '
                    "Fix provided repair function to properly repair parameters."
                )

        self.called = True

        return repaired


def repair(
    *constraint_definition,
    name: str = None,
    repair_function: Callable[[Matrix], Matrix] = None,
    check_repaired: str = _CHECK_ALWAYS,
) -> Repair:
    """Repair factory function.
    Constructs a repair function for a set membership constraint.

    Args:
        constraint_definition: list of arguments defining the constraint.
        name: optional constraint name.

            - Passed to the constraint function.
            - Ignored if a Constraint object is provided.

        repair_function: function used to repair solutions.
        check_repaired: repair function strictness. Checking methods are
        defined as "never" to never check that the repair function correctly
            "once" to only check that the repair function correctly repaired once on
            the first time its called, and "always" to always check that the  repair
            function correctly repaired the solutions.

    Returns:
        Repair object with expected properties.

    Raises:
        ValueError: if left-hand side of constraint definition is not a string or the
            partial returned by optimizer.utils.functional.column.
        ValueError: if the user specifies a non-set membership constraint and does not
            provide a repair function.
    """
    constraint_ = constraint(*constraint_definition, name=name)

    if isinstance(constraint_, SetMembershipConstraint) and repair_function is None:
        if (
            not hasattr(constraint_.constraint_func, "keywords")
            or "col" not in constraint_.constraint_func.keywords
        ):
            raise ValueError(
                "Left-hand side of a set-based repair must either be a string "
                "or the partial returned by optimizer.utils.functional.column."
            )

        repair_ = SetRepair(constraint_)

    else:
        if repair_function is None:
            raise ValueError(
                "A repair function must be provided if repairing a constraint that is "
                "not a set membership constraint."
            )

        repair_ = UserDefinedRepair(
            constraint_,  # noqa
            repair_function=repair_function,
            check_repaired=check_repaired,
        )

    return repair_
