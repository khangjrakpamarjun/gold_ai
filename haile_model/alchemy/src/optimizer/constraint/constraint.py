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
Constraint module.
"""

import abc
from itertools import starmap
from numbers import Real
from typing import Any, Callable, List, Union

import numpy as np
import pandas as pd

from optimizer.constraint.sets import ConstraintSet, UserDefinedSet
from optimizer.constraint.utils import ConstantCallable
from optimizer.exceptions import InvalidConstraintError
from optimizer.types import Matrix, Predictor, Vector
from optimizer.utils.functional import column
from optimizer.utils.validation import check_matrix

_LESS_THAN = "<="
_GREATER_THAN = ">="
_EQUAL_TO = "=="
_SET_MEMBERSHIP = "in"

_INEQUALITIES = _GREATER_THAN + _LESS_THAN
_COMPARATORS = _LESS_THAN + _GREATER_THAN + _EQUAL_TO + _SET_MEMBERSHIP


class BaseConstraint(abc.ABC):
    """
    Represents a constraint.
    """

    def __init__(
        self,
        constraint_func: Union[Callable[[Matrix], Vector], Predictor],
        name: str = None,
    ):
        """Constructor.

        Args:
            constraint_func: callable or predictor that accepts a matrix and returns
            a vector.
            name: optional name for logging.

        Raises:
            InvalidConstraintError: when constraint_func is not callable or does not
            have a callable predict method.
        """
        if not callable(constraint_func):
            if not hasattr(constraint_func, "predict") or not callable(
                constraint_func.predict
            ):
                raise InvalidConstraintError(
                    "Constraint value function must be callable "
                    "or have a callable predict method."
                )

        self._constraint_func = constraint_func
        self.name = name

        self.constraint_values = None
        self.distances = None

    @property
    def constraint_func(self) -> Any:
        """Resolved constraint function"""
        return self._resolve_attr(self._constraint_func)

    @abc.abstractmethod
    def calculate_distances(
        self,
        constraint_eval: Vector,
        parameters: Matrix,
        save_eval: bool = False,
    ) -> Vector:
        """Calculate the distances of the evaluated constraint from
        the constraint's boundaries.

        Args:
            constraint_eval: Vector, result of calling constraint_func.
            parameters: Matrix, parameters to evaluate.
                May be required to evaluate constraint boundaries defined by functions.
            save_eval: True to save evaluation of constraint sides.

        Returns:
            Nonnegative Vector of distances from the constraint's boundaries.
        """

    @property
    def violated(self) -> Vector:
        """Determine which rows of the previously given parameter matrix violated
        the constraint.

        Returns:
            Boolean Vector.

        Raises:
            AttributeError: when the constraint has not yet been evaluated.
        """
        if self.constraint_values is None:
            raise AttributeError(
                "Attempted to determine where a constraint was "
                "violated without first evaluating."
            )

        # Allow for rounding error in greater than 0 comparison.
        return self.distances > np.sqrt(np.finfo(float).eps)

    def __call__(self, parameters: Matrix) -> Vector:
        """Evaluate the constraint on the given parameter Matrix.

        Args:
            parameters: Matrix, parameters to evaluate.

        Returns:
            Vector of distances from the constraint boundary.
        """
        check_matrix(parameters)

        self.constraint_values = self.constraint_func(parameters)
        self.distances = self.calculate_distances(
            self.constraint_values, parameters, save_eval=True
        )

        return self.distances

    @staticmethod
    def _resolve_attr(obj: Any) -> Any:
        """
        Returns an object's predict method if
        available or the object itself.
        """
        if hasattr(obj, "predict"):
            return obj.predict
        return obj


class InequalityConstraint(BaseConstraint):
    """
    Represents an non-strict inequality constraint.
    """

    def __init__(
        self,
        constraint_func: Union[Callable[[Matrix], Vector], Predictor],
        lower_bound: Union[Real, Callable[[Matrix], Vector], Predictor] = None,
        upper_bound: Union[Real, Callable[[Matrix], Vector], Predictor] = None,
        name: str = None,
    ):
        """Constructor.

        Args:
            constraint_func: calculates the value of the constraint somehow.
                             For example, if our constraint is: 0 <= 4x_1 + x_2 <= 3
                             This object will return 4x_1 + x_2.
                             - Callable: will be called directly in the evaluation.
                             - Predictor: predict method will be used
                             instead of calling.
            lower_bound: lower bound of the constraint.
                         - Real: constant value for the lower bound.
                         - Callable: will be called before comparing to inputs.
                         - Predictor: same as callable but using predict method.
            upper_bound: upper bound of the constraint.
                         - Real: constant value for the lower upper.
                         - Callable: will be called before comparing to inputs.
                         - Predictor: same as callable but using predict method.
            name: optional name for logging.

        Raises:
            InvalidConstraintError: if one of lower_bound or upper_bound is not given.
        """
        super(InequalityConstraint, self).__init__(constraint_func, name=name)

        # Must provide some sort of constraint.
        if lower_bound is None and upper_bound is None:
            raise InvalidConstraintError(
                "A lower and/or upper bound or an "
                "equality constraint must be specified."
            )

        self._lower_bound = np.NINF if lower_bound is None else lower_bound
        self._upper_bound = np.PINF if upper_bound is None else upper_bound

        # Values for logging.
        self.lower_bound_eval = None
        self.upper_bound_eval = None

    @property
    def upper_bound(self) -> Any:
        """Resolved upper bound"""
        return self._resolve_attr(self._upper_bound)

    @property
    def lower_bound(self) -> Any:
        """Resolved lower bound"""
        return self._resolve_attr(self._lower_bound)

    def calculate_distances(
        self, constraint_eval: Vector, parameters: Matrix, save_eval: bool = False
    ) -> Vector:
        """Calculates distance from upper and lower bound.

        Args:
            constraint_eval: Vector, result of calling constraint_func.
            parameters: Matrix, parameters to evaluate.
                May be required to evaluate constraint boundaries defined by functions.
            save_eval: True to save evaluation of constraint sides.

        Returns:
            Vector of distances from the constraint's boundaries.
                - Values where the constraint is satisfied will be nonpositive.
        """
        lower_bound_eval = (
            self.lower_bound
            if not callable(self.lower_bound)
            else self.lower_bound(parameters)  # pylint: disable=not-callable
        )

        if save_eval:
            self.lower_bound_eval = lower_bound_eval

        upper_bound_eval = (
            self.upper_bound
            if not callable(self.upper_bound)
            else self.upper_bound(parameters)  # pylint: disable=not-callable
        )

        if save_eval:
            self.upper_bound_eval = upper_bound_eval

        # Distance to nearest constraint.
        return np.maximum(
            lower_bound_eval - constraint_eval, constraint_eval - upper_bound_eval
        )


class EqualityConstraint(BaseConstraint):
    """
    Represents an equality constraint.
    """

    def __init__(
        self,
        constraint_func: Union[Callable[[Matrix], Vector], Predictor],
        equality: Union[Real, Callable[[Matrix], Vector], Predictor] = None,
        equality_epsilon: float = 1e-6,
        name: str = None,
    ):
        """Constructor.

        Args:
            constraint_func: calculates the value of the constraint somehow.
                             For example, if our constraint is: x_1^2 + x_2 == 10
                             This object will return x_1^2 + x_2.
                             - Callable: will be called directly in the evaluation.
                             - Predictor: predict method will be used
                             instead of calling.
            equality: equality constraint specifier.
                      - Real: constant value for the equality.
                      - Callable: will be called before comparing to inputs.
                      - Predictor: same as callable but using predict method.
            equality_epsilon: float, small number for float comparisons.
            name: optional name for logging.
        """
        super(EqualityConstraint, self).__init__(constraint_func, name=name)

        self._equality = equality
        self.equality_epsilon = equality_epsilon

        self.equality_eval = None

    @property
    def equality(self) -> Any:
        """Resolved equality"""
        return self._resolve_attr(self._equality)

    def calculate_distances(
        self, constraint_eval: Vector, parameters: Matrix, save_eval: bool = False
    ) -> Vector:
        """Calculates distance from a particular equality value.

        Args:
            save_eval:
            constraint_eval: Vector, result of calling constraint_func.
            parameters: Matrix, parameters to evaluate.
                May be required to evaluate constraint boundaries defined by functions.
            save_eval: True to save evaluation of constraint sides.

        Returns:
            Nonnegative Vector of distances from the constraint's equality value.
        """
        equality_eval = (
            self.equality
            if not callable(self.equality)
            else self.equality(parameters)  # pylint: disable=not-callable
        )

        if save_eval:
            self.equality_eval = equality_eval

        distances = np.abs(constraint_eval - equality_eval)
        distances[distances <= self.equality_epsilon] = 0

        return distances


class SetMembershipConstraint(BaseConstraint):
    """
    Represents a set membership constraint.
    """

    def __init__(
        self,
        constraint_func: Union[Callable[[Matrix], Vector], Predictor],
        constraint_set: Union[ConstraintSet, List, Vector],
        name: str = None,
    ):
        """Constructor.

        Args:
            constraint_func: calculates the value of the constraint somehow.
                             For example, if our constraint is:
                             x_1 * x_2  in  [1, 2, 3, 4]
                             This object will return x_1 * x_2.
                             - Callable: will be called directly in the evaluation.
                             - Predictor: predict method will be used
                             instead of calling.
            constraint_set: the set describing the constraint.
                            - ConstraintSet: uses the nearest method of a
                            constraint set.
                            - Vector/List: constructs a UserDefinedSet and applies the
                            nearest method.
            name: optional name for logging.
        """
        super(SetMembershipConstraint, self).__init__(constraint_func, name=name)

        if isinstance(constraint_set, (list, pd.Series, np.ndarray)):
            constraint_set = UserDefinedSet(constraint_set)

        self.constraint_set = constraint_set

    def calculate_distances(
        self, constraint_eval: Vector, parameters: Matrix, save_eval: bool = False
    ) -> Vector:
        """Calculates distance from the nearest value in the provided set.

        Args:
            constraint_eval: Vector, result of calling constraint_func.
            parameters: Matrix, parameters to evaluate. May be required to
                        evaluate constraint boundaries defined by functions.
            save_eval: True to save evaluation of constraint sides (not used).

        Returns:
            Nonnegative Vector of distances from the constraint's equality value.
        """
        return self.constraint_set.distance(constraint_eval)


def one_sided_constraint(
    lhs: Union[Real, Callable[[Matrix], Vector], Predictor, str],
    comp: str,
    rhs: Union[
        Real, Callable[[Matrix], Vector], Predictor, ConstraintSet, Vector, List, str
    ],
    equality_epsilon: float = 1e-6,
    name: str = None,
) -> BaseConstraint:
    """Constraint factory function.
    Constructs the object for a single sided constraint.

    Args:
        lhs: left hand side of constraint.
             - Real: constant value for the lower bound.
             - Callable: will be called before comparing to inputs.
             - Predictor: same as callable but using predict method.
             - str: used to index a passed DataFrame.
        comp: str, comparator or "in" for set inclusion logic.
        rhs: right hand side of constraint.
             - Real: constant value for the lower bound.
             - Callable: will be called before comparing to inputs.
             - Predictor: same as callable but using predict method.
             - ConstraintSet: special sets to use for set membership (e.g. Integers).
             - Vector, List: user defined set for set membership constraint.
             - str: used to index a passed DataFrame.
        equality_epsilon: float, small number for float comparisons.
        name: optional constraint name.

    Returns:
        Constraint object with the expected properties.

    Raises:
        InvalidConstraintError: when at least one side of the constraint is not a
                                callable or an object with a predict method.
        ValueError: when the provided comparator is invalid.
        ValueError: if user specifies a set-membership constraint lhs is a number.
        ValueError: if user specifies a set-membership constraint and rhs is not a
                    list or ConstraintSet.
    """
    lhs = column(lhs) if isinstance(lhs, str) else lhs
    rhs = column(rhs) if isinstance(rhs, str) else rhs

    # At least one side of the constraint must be callable.
    if not any(
        [callable(lhs), callable(rhs), hasattr(lhs, "predict"), hasattr(rhs, "predict")]
    ):
        raise InvalidConstraintError(
            f"At least one side of your constraint must be a callable or "
            f"an object with a predict method. "
            f"Given types {type(lhs).__name__} and {type(rhs).__name__}."
        )

    comp = comp.strip()

    if comp not in _COMPARATORS:
        raise ValueError(
            f'Invalid comparator "{comp}" provided. '
            f"""Valid comparators are: {", ".join(_COMPARATORS)}"""
        )

    # Left hand side is always passed as the constraint function.
    # As a result, it must be callable. In the case that its a constant,
    # we can just wrap it as a function that always returns itself as an array.
    if not callable(lhs) and not hasattr(lhs, "predict"):
        lhs = ConstantCallable(lhs)

    if comp in _LESS_THAN + _GREATER_THAN:
        lower_bound, upper_bound = None, None

        if comp in _GREATER_THAN:
            lower_bound = rhs

        else:
            upper_bound = rhs

        constraint_ = InequalityConstraint(
            lhs, lower_bound=lower_bound, upper_bound=upper_bound, name=name
        )

    elif comp in _EQUAL_TO:
        constraint_ = EqualityConstraint(
            lhs, rhs, equality_epsilon=equality_epsilon, name=name
        )

    else:  # Set membership constraint.
        if isinstance(lhs, Real):
            raise ValueError(f"Nonsensical constraint provided: {lhs} in {rhs}.")

        if not isinstance(rhs, (ConstraintSet, list)):
            raise ValueError(
                "For set membership constraints, right hand side must be "
                "a ConstraintSet or list of values."
            )

        constraint_ = SetMembershipConstraint(lhs, rhs, name=name)

    return constraint_


def two_sided_constraint(
    lhs: Union[Real, Callable[[Matrix], Vector], Predictor, str],
    left_comp: str,
    middle: Union[Real, Callable[[Matrix], Vector], Predictor, str],
    right_comp: str,
    rhs: Union[Real, Callable[[Matrix], Vector], Predictor, str],
    name: str = None,
) -> BaseConstraint:
    """Constraint factory function.
    Constructs the object for a two-sided constraint.

    Inequalities of the form x >= y >= z will be converted to z <= y <= x due to
    `InequalityConstraint` expecting a lower/upper bound and constraint function.

    Args:
        lhs: left-hand side of constraint.
             - Real: constant value for the lower bound.
             - Callable: will be called before comparing to inputs.
             - Predictor: same as callable but using predict method.
             - str: used to index a passed DataFrame.
        left_comp: comparator.
        middle: middle of constraint.
             - Real: constant value for the lower bound.
             - Callable: will be called before comparing to inputs.
             - Predictor: same as callable but using predict method.
             - str: used to index a passed DataFrame.
        right_comp: comparator.
        rhs: right-hand side of constraint.
             - Real: constant value for the lower bound.
             - Callable: will be called before comparing to inputs.
             - Predictor: same as callable but using predict method.
             - str: used to index a passed DataFrame.
        name: optional constraint name.

    Returns:
        Constraint object with the expected properties.

    Raises:
        InvalidConstraintError: if none of the sides of the constraint are callable or
            objects with predict methods.
        ValueError: if using a non-inequality comparator.
        ValueError: if comparators are not equal.

    """
    lhs = column(lhs) if isinstance(lhs, str) else lhs
    middle = column(middle) if isinstance(middle, str) else middle
    rhs = column(rhs) if isinstance(rhs, str) else rhs

    if not any(map(callable, [lhs, middle, rhs])) and not any(
        starmap(hasattr, [(lhs, "predict"), (middle, "predict"), (rhs, "predict")])
    ):
        raise InvalidConstraintError(
            f"At least one side of your constraint must be a callable or "
            f"an object with a predict method. "
            f"Given types {type(lhs).__name__}, {type(middle).__name__}, "
            f"and {type(rhs).__name__}."
        )

    right_comp, left_comp = right_comp.strip(), left_comp.strip()

    if left_comp not in _INEQUALITIES:
        raise ValueError(f"Left comparator must be an inequality. Given {left_comp}.")

    if right_comp not in _INEQUALITIES:
        raise ValueError(f"Right comparator must be an inequality. Given {right_comp}.")

    if left_comp != right_comp:
        raise ValueError(
            f"Comparators must be the same. "
            f"Given {left_comp} and {right_comp}. "
            f"Convert constraints of the form lhs >= middle <= rhs "
            f"to one sided constraints of the form middle <= min(lhs, rhs)."
        )

    # The middle side is always passed as the constraint function.
    # As a result, it must be callable. In the case that it is a constant,
    # we can just wrap it as a function that always returns itself as an array.
    if not callable(middle) and not hasattr(middle, "predict"):
        middle = ConstantCallable(middle)

    # InequalityConstraint expects constraints of the form x <= y <= z.
    if left_comp in _GREATER_THAN:
        lhs, rhs = rhs, lhs

    return InequalityConstraint(middle, lower_bound=lhs, upper_bound=rhs, name=name)


def constraint(
    *constraint_definition, equality_epsilon: float = 1e-6, name: str = None
) -> BaseConstraint:
    """Constraint factory function.
    Passes the constraint definition to the expected functions.

    Usage:
        Inequality::

            constraint(lambda x: x["column"] ** 2, "<=", 10)
            constraint(lambda x: x["column"], ">=", lambda x: x["other_column"])
            constraint(my_model, "<=", 4500)
            constraint(0, "<=", lambda x: x["b"], "<=", 100)
            constraint(lambda x: x["a"], ">=", lambda x: x["b"], ">=", lambda x: x["c"])

        Equality::

            constraint(lambda x: np.sum(x[:, [1, 2, 4]], axis=1), "==", 30)
            constraint(1, "==", my_model)

        Set membership::

            constraint(lambda x: x[:, 0], "in", MultiplesOf(3))
            constraint(lambda x: x[:, 3], "in", Integers())
            constraint(lambda x: x[:, 4], "in", [1, 2, 3, 5, 8, 13])

    Args:
        *constraint_definition: list of arguments that define the constraint.
        equality_epsilon: float, small number for float comparisons.
        name: str, optional name of constraint.

    Returns:
        Constraint object with the expected properties.
    """
    if len(constraint_definition) == 1:
        constraint_ = constraint_definition[0]

        if not isinstance(constraint_, BaseConstraint):
            raise InvalidConstraintError(
                "Length 1 constraint definitions must be an instance of BaseConstraint."
            )

    elif len(constraint_definition) == 3:
        constraint_ = one_sided_constraint(
            *constraint_definition, equality_epsilon=equality_epsilon, name=name
        )

    elif len(constraint_definition) == 5:
        constraint_ = two_sided_constraint(*constraint_definition, name=name)

    else:
        raise InvalidConstraintError(
            f"Constraint definitions must be of length 1, 3, or 5. "
            f"Given length {len(constraint_definition)}."
        )

    return constraint_
