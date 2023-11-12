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
Best solution logger module.
"""

import numpy as np
import pandas as pd

from optimizer.loggers.base import LoggerMixin
from optimizer.solvers import Solver
from optimizer.types import Matrix, Vector


class BestTrajectoryLogger(LoggerMixin):
    """
    Logs best solution and objective value for each iteration

    Examples:
        Using the logger::

            >>> logger = BestTrajectoryLogger()
            >>> parameters = solver.ask()
            >>> objectives, repaired_parameters = problem(parameters)
            >>> solver.tell(repaired_parameters, objectives)
            >>> logger.log(solver)  # Exact inputs will differ between loggers.
    """

    def __init__(self):
        """Constructor."""
        self.data_dict = {}
        self.iteration = 0

    def log(self, solver: Solver):  # pylint: disable=arguments-differ
        """Append best solution and objective value to the data dictionary.

        Args:
            solver: Solver object.
        """
        solution, objective_value = solver.best()

        self.data_dict[self.iteration] = {
            "solution": solution,
            "objective_value": objective_value,
        }
        self.iteration += 1

    @property
    def log_df(self) -> pd.DataFrame:
        """Return a DataFrame of the stored dictionary values.

        Returns:
            pd.DataFrame.
        """
        df = pd.DataFrame.from_dict(self.data_dict, orient="index")
        df.index = df.index.rename("iteration")

        return df

    @property
    def objective_values(self) -> Vector:
        return self.log_df["objective_value"].values

    @property
    def solutions(self) -> Matrix:
        return np.stack(self.log_df["solution"])
