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
Basic File-based Logger module.
"""
import numpy as np
import pandas as pd

from optimizer.loggers.base import LoggerMixin
from optimizer.loggers.base_file import FileLoggerBase
from optimizer.solvers import Solver


class BasicFileLogger(LoggerMixin, FileLoggerBase):
    """
    Logs summary statistics about an optimization.

    File-based logging requires installation of Tensorflow.
    Examples:
        Using the BasicLogger to write to <path_to_update>. \
        Tensorboard can then be called from terminal via \
        `tensorboard --logdir <log_path>`::

            >>> logger = BasicFileLogger(log_path=f'logs/basic_logger/{path_to_update}')
            >>> parameters = solver.ask()
            >>> objectives, repaired_parameters = problem(parameters)
            >>> solver.tell(repaired_parameters, objectives)
            >>> logger.log(solver)  # Exact inputs will differ between loggers.
    """

    def __init__(self, log_path=None):
        """
        Constructor.

        Args:
            log_path: Path to save TensorBoard Logs
        """
        super().__init__(log_path)
        self.data_dict = {}
        self.iteration = 0

    def log(self, solver: Solver):  # pylint: disable=arguments-differ
        """Append logged values to the data dictionary.
        Rather than appending to a DataFrame, this adds an entry to a dictionary
        which is much faster in practice.

        Args:
            solver: Solver object.
        """
        objective_values = solver.objective_values

        min_, q25, med, q75, max_ = np.quantile(
            objective_values, [0, 0.25, 0.5, 0.75, 1]
        )

        self.data_dict[self.iteration] = {
            "mean_objective": np.mean(objective_values),
            "std_objective": np.std(objective_values),
            "median_objective": med,
            "25th_quartile_objective": q25,
            "75th_quartile_objective": q75,
            "min_objective": min_,
            "max_objective": max_,
        }
        self.write_array_to_disk(
            objective_values, self.iteration, name="Objective/Trace"
        )
        self.iteration += 1

    @property
    def log_df(self) -> pd.DataFrame:
        """Return a DataFrame of the stored dictionary values.

        Returns:
            pd.DataFrame
        """
        df = pd.DataFrame.from_dict(self.data_dict, orient="index")
        df.index = df.index.rename("iteration")

        df["overall_min_objective"] = df["min_objective"].expanding().min()
        df["overall_max_objective"] = df["max_objective"].expanding().max()

        return df
