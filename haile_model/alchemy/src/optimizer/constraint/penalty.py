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
Penalty module.
"""

from numbers import Real
from typing import Callable, Union

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from optimizer.constraint.constraint import BaseConstraint, constraint
from optimizer.constraint.handler import BaseHandler
from optimizer.constraint.utils import ConstantCallable
from optimizer.exceptions import InvalidConstraintError
from optimizer.types import Matrix, Vector

from . import penalties


class Penalty(BaseHandler):
    """
    Represents a penalty function.
    """

    def __init__(
        self,
        constraint_: BaseConstraint,
        penalty_function: Union[str, Callable[[Matrix], Vector], Real] = "linear",
        penalty_multiplier: Real = 1.0,
    ):
        """Constructor.

        Args:
            constraint_: BaseConstraint object representing the
                         constraint being penalized.
            penalty_function: how to calculate the penalty from the constraint value.
                              - str: uses a builtin method.
                              -- 'linear': penalty scales with absolute distance.
                              -- 'quadratic': penalty scales quadratically
                              with distance.
                              - callable: a function to calculate a custom penalty.
                              This function will accept a Vector of distances from the
                              boundaries of the constraint and return a Vector of
                              the same shape.
                              - Real: constant penalty for violating the constraint.
            penalty_multiplier: Real, constant to multiply the penalty.

        Raises:
            AttributeError: if the provided penalty function string is not defined.
            InvalidConstraintError: if the penalty function is not a valid type.
        """
        super(Penalty, self).__init__(constraint_)

        if isinstance(penalty_function, str):
            try:
                self.penalty = getattr(penalties, penalty_function)
            except AttributeError:
                raise InvalidConstraintError(
                    f"{penalty_function} is an invalid built-in penalty function. "
                    f"See Penalty docstring for valid built-ins."
                )

        elif callable(penalty_function):
            self.penalty = penalty_function

        elif isinstance(penalty_function, Real):
            self.penalty = ConstantCallable(penalty_function)

        else:
            raise InvalidConstraintError(
                "Penalty argument must be a string, callable, or real number."
            )

        self.penalty_multiplier = penalty_multiplier

        # Stored values of penalty evaluation for logging.
        self.calculated_penalty = None

    def __call__(self, parameters: Matrix) -> Vector:
        """Evaluate the constraint and calculate penalty values.

        Args:
            parameters: Matrix, parameters to evaluate.

        Returns:
            Vector, of penalty values. Values of 0 represent a satisfied constraint.
        """
        distances = self.constraint(parameters)

        self.calculated_penalty = self.penalty_multiplier * self.penalty(distances)

        # Only apply penalty where the constraint was violated.
        self.calculated_penalty[~self.constraint.violated] = 0

        return self.calculated_penalty.astype(float)


def penalty(
    *constraint_definition,
    penalty_function: Union[str, Callable[[Matrix], Vector], Real] = "linear",
    penalty_multiplier: Real = 1.0,
    name: str = None,
    equality_epsilon: float = 1e-6,
) -> Penalty:
    """Penalty factory function.
    Constructs a the penalty function for a single sided constraint.

    Usage:
        Inequality::

            penalty(lambda x: x["column"] ** 2, "<=", 10)
            penalty(lambda x: x["column"], ">=", lambda x: x["other_column"])
            penalty(my_model, "<=", 4500)

        Equality::

            penalty(lambda x: np.sum(x[:, [1, 2, 4]], axis=1), "==", 30)
            penalty(1, "==", my_model)

        Set membership::

            penalty(lambda x: x[:, 0], "in", MultiplesOf(3))
            penalty(lambda x: x[:, 3], "in", Integers())
            penalty(lambda x: x[:, 4], "in", [1, 2, 3, 5, 8, 13])

    Or a BaseConstraint object may simply be specified:
        penalty(my_constraint)

    Args:
        constraint_definition: list of arguments that define the constraint.
        penalty_function: typing.Union[str, Callable, Real], how to calculate
            the penalty from the constraint value.

            - str: uses a builtin method.
                - 'linear': penalty scales with absolute distance.
                - 'quadratic': penalty scales quadratically with distance.

            - typing.Callable: a function to calculate a custom penalty. This
              function will accept a Vector of distances from the boundaries of
              the constraint and return a Vector of the same shape.
            - Real: constant penalty for violating the constraint.

        penalty_multiplier: Real, constant to multiply the penalty.
        name: str, optional name for logging.

            - Passed to the constraint function.
            - Ignored if a Constraint object is provided.
        equality_epsilon: float, small number for float comparisons.

    Returns:
        Penalty object with the expected properties.
    """
    constraint_ = constraint(
        *constraint_definition, equality_epsilon=equality_epsilon, name=name
    )

    return Penalty(
        constraint_,
        penalty_function=penalty_function,
        penalty_multiplier=penalty_multiplier,
    )


class _ScoreFunctionWrapper:
    """Helper class to make the history penalty serializable."""

    def __init__(self, score_func: callable, pipeline: Pipeline):
        self.score_func = score_func
        self.pipeline = pipeline

    def __call__(self, matrix: Matrix) -> Vector:
        return self.score_func(self.pipeline.transform(matrix))


def history_penalty(
    historical_data: Matrix,
    penalty_multiplier: Real = 1.0,
    pipeline: Pipeline = None,
    name: str = None,
    score_func: Callable[[Matrix], Vector] = None,
) -> Penalty:
    """History penalty factory: this function creates a Penalty object that based on
    historical data will penalize solutions that are far from historical values.

    By adjusting the penalty_multiplier one can explicitly balance the trade-off
    between staying close to historical ranges and exploring the broader search space.

    By using the pipeline one can restrict the dimensions taken into account for
    historical deviations. This is useful if we're optimizing a 60-dimensional problem
    but in one part of the operation we're stumbling into non-clear constraints, and
    these constraints only are related to say 5 dimensions. By restricting our penalty
    to those 5 dimensions we won't suffer the course of dimensionality. Also, if we have
    some feature engineering in our modeling pipeline, we can pass this pipeline as
    parameter in order to look for anomalies on the engineered columns.

    Args:
        historical_data: Matrix containing the historical values
        pipeline: A fit sklearn pipeline that has a set of Transformer objects
        penalty_multiplier: Real, the value by which the anomaly score is multiplied
        name: str, name of the penalty
        score_func: callable, optional, if provided will use this callable for
         computing the score with which a solution will be penalized. This callable
         has to return values < 0 if a penalty is to be applied. If a value >= 0 is
         returned, then no penalty will be applied (this is compliant with the interface
         of sklearn.ensemble.IsolationForest.decision_function). Also, note that if a
         pipeline is given as a parameter, then this function will receive whatever
         the output of pipeline.transform is. The signature of this callable must be
         Matrix -> Vector.

    Returns:
        Penalty

    """
    if name is None:
        name = "history"

    if pipeline is None:
        pipeline = Pipeline([("identity", None)])

    if score_func is None:
        clf = IsolationForest(n_jobs=-1)
        clf.fit(pipeline.transform(historical_data))
        score_func = clf.decision_function

    lhs = _ScoreFunctionWrapper(score_func, pipeline)

    return penalty(lhs, ">=", 0, penalty_multiplier=penalty_multiplier, name=name)
