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
Penalty logger package.
"""

from typing import List, Union

from optimizer.constraint import Penalty
from optimizer.constraint.constraint import InequalityConstraint
from optimizer.loggers.base import LoggerMixin
from optimizer.problem import OptimizationProblem


class PenaltyLogger(LoggerMixin):
    """
    Logs information about penalties.
    """

    def __init__(self):
        """Constructor."""
        self.data_dict = {}
        self.iteration = 0

    def log(  # pylint: disable=arguments-differ
        self, penalties: Union[List[Penalty], OptimizationProblem, Penalty]
    ):
        """Log information about penalties.
        Penalties should have been evaluated before passing to this method.

        Args:
            penalties: a Penalty, list of Penalties,
                or an OptimizationProblem object.

        Raises:
            RuntimeError: when logging a penalty that has not been evaluated.
        """
        if isinstance(penalties, OptimizationProblem):
            penalties = penalties.penalties

        if isinstance(penalties, Penalty):
            penalties = [penalties]

        data = {}

        for i, penalty in enumerate(penalties):

            if penalty.calculated_penalty is None:
                raise RuntimeError("Attempted to log an unevaluated constraint.")

            name = f"penalty_{i}" if penalty.name is None else penalty.name
            constraint = penalty.constraint

            penalty_data = {}

            #
            # If upper and lower bounds are some kind of function,
            # their values will be useful and should be logged too.
            # Otherwise, just log the single constant.
            #
            if isinstance(constraint, InequalityConstraint):
                lower = (
                    constraint.lower_bound
                    if not callable(constraint.lower_bound)
                    else constraint.lower_bound_eval
                )
                upper = (
                    constraint.upper_bound
                    if not callable(constraint.upper_bound)
                    else constraint.upper_bound_eval
                )

                penalty_data["lower_bound"] = lower
                penalty_data["upper_bound"] = upper

            penalty_data["value"] = list(constraint.constraint_values)
            penalty_data["penalty"] = list(penalty.calculated_penalty)

            data[name] = penalty_data

        self.data_dict[self.iteration] = data
        self.iteration += 1
