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
Penalty File-logger package.
"""
from typing import List, Union

from optimizer.constraint import Penalty
from optimizer.constraint.constraint import InequalityConstraint
from optimizer.loggers.base import LoggerMixin
from optimizer.loggers.base_file import FileLoggerBase
from optimizer.problem import OptimizationProblem


class PenaltyFileLogger(LoggerMixin, FileLoggerBase):
    """
    Logs information about penalties.


    Examples:
        Using the penalty Logger and writing to <path_to_update>::

            >>> logger = PenaltyFileLogger(log_path=f'logs/penalty/{path_to_update}')
            >>> parameters = solver.ask()
            >>> objectives, repaired_parameters = problem(parameters)
            >>> solver.tell(repaired_parameters, objectives)
            >>> logger.log(problem)  # Exact inputs will differ between loggers.

    """

    def __init__(self, log_path=None):
        """
        Constructor.
        File-based logging requires installation of Tensorflow

        Args:
            log_path: Optional, path to save tensorboard logs.
        """
        super().__init__(log_path)
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
                self.write_array_to_disk(lower, self.iteration, f"{name}-lower_bound")
                self.write_array_to_disk(upper, self.iteration, f"{name}-upper_bound")

            penalty_data["value"] = list(constraint.constraint_values)
            penalty_data["penalty"] = list(penalty.calculated_penalty)

            self.write_array_to_disk(
                constraint.constraint_values, self.iteration, f"{name}-value"
            )
            self.write_array_to_disk(
                penalty.calculated_penalty, self.iteration, f"{name}-calculated_penalty"
            )

            data[name] = penalty_data

        self.data_dict[self.iteration] = data
        self.iteration += 1
