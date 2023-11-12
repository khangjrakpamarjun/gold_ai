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
Holds definitions for the StatefulOptimizationProblem
"""

from copy import deepcopy
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from optimizer.constraint.penalty import Penalty
from optimizer.constraint.repair import Repair
from optimizer.problem.problem import OptimizationProblem
from optimizer.types import Matrix, Vector
from optimizer.utils.functional import safe_subset
from optimizer.utils.validation import check_matrix


class StatefulOptimizationProblem(OptimizationProblem):
    """
    Optimization problem representing the case where both optimizable and
    non-optimizable parameters must be handled.
    """

    def __init__(
        self,
        objective: Callable[[Matrix], Vector],
        state: Union[Vector, Matrix],
        optimizable_columns: Union[List, pd.Series],
        penalties: Union[
            Penalty,
            Callable[[Matrix], Vector],
            List[Union[Penalty, Callable[[Matrix], Vector]]],
        ] = None,
        repairs: Union[
            Repair,
            Callable[[Matrix], Matrix],
            List[Union[Repair, Callable[[Matrix], Matrix]]],
        ] = None,
        sense: str = "minimize",
    ):
        """Constructor.

        Args:
            objective: callable object representing the objective function.
            state: Vector representing the current state of the problem.
            optimizable_columns: list of columns/dimensions that can be optimized.
            penalties: optional Penalty or list of Penalties.
            repairs: optional Repairs or callable or list of Repairs and/or callables.
            sense: 'minimize' or 'maximize', how to optimize the objective.
        """
        super(StatefulOptimizationProblem, self).__init__(
            objective, penalties=penalties, repairs=repairs, sense=sense
        )
        # An internal current state will save us time for any processing
        # we might need to apply to the current state before subbing in
        # the optimizable variables in the __call__ method.
        self.optimizable_columns = list(optimizable_columns)
        self._state = None
        self.state = state

    @property
    def state(self) -> Vector:
        """
        Returns the current (actual) state

        Returns:
            vector representing the current state of the problem
        """
        return self._state

    @state.setter
    def state(self, new_state: Union[Matrix, Vector]):
        """
        Set actual state and reset internal state. The new state can be
        one of three things:
            - a single-row dataframe
            - a pandas series with a numeric dtype
            - a numpy array
        pandas series with other dtypes are not allowed to protect users from
        potential dtype errors that arrise when a dataframe's row is converted
        into a series.

        Args:
            new_state: vector representing the new state of the problem

        Raises:
            TypeError: in case of wrong input type
        """
        new_state = deepcopy(new_state)

        type_error_msg = (
            "`state` must be a single-row dataframe, "
            "a numeric series, or a numpy array. "
            "Got {}."
        )

        if isinstance(new_state, pd.Series):
            if not is_numeric_dtype(new_state):
                raise TypeError(
                    type_error_msg.format(f"a series of type {new_state.dtype}")
                )
            new_state = new_state.to_frame().T
            all_columns = new_state.columns.tolist()
        elif isinstance(new_state, pd.DataFrame):
            if not len(new_state) == 1:
                raise TypeError(
                    type_error_msg.format(f"a dataframe with {len(new_state)} rows")
                )
            all_columns = new_state.columns.tolist()
        elif isinstance(new_state, np.ndarray):
            all_columns = list(np.arange(len(new_state)))
        else:
            raise TypeError(
                type_error_msg.format(f"an object of type {str(type(new_state))}")
            )

        # Check that the new state contains the given columns
        difference = set(self.optimizable_columns) - set(all_columns)
        if difference:
            raise KeyError(
                f"Provided optimizable columns not found in provided state: "
                f"{sorted(list(difference))}"
            )
        self._state = new_state

    @property
    def non_optimizable_columns(self) -> List[Union[str, int]]:
        """Get a list of columns that are not being optimized.

        Returns:
            List of ints or strings.
        """
        if isinstance(self.state, np.ndarray):
            columns = set(range(len(self.state))) - set(self.optimizable_columns)

        else:
            columns = set(self.state.columns) - set(self.optimizable_columns)

        return sorted(list(columns))

    def __call__(
        self,
        parameters: Matrix,
        skip_penalties: bool = False,
        skip_repairs: bool = False,
    ) -> Tuple[Vector, Matrix]:
        """Evaluate the OptimizationProblem on a Matrix of parameters.

        Args:
            parameters: Matrix of parameters to evaluate.
            skip_penalties: if True, evaluate without applying penalties.
            skip_repairs: if True, evaluate without applying repairs.

        Returns:
            Vector of objective values and (possibly repaired) parameter matrix.
        """
        check_matrix(parameters)

        params_plus_state = self.substitute_parameters(parameters)

        objectives, params_plus_state = super(
            StatefulOptimizationProblem, self
        ).__call__(params_plus_state, skip_penalties, skip_repairs)

        return objectives, safe_subset(params_plus_state, self.optimizable_columns)

    def substitute_parameters(self, parameters: Matrix) -> Matrix:
        """Substitute the optimizable parameters into the current state.

        Args:
            parameters: Matrix of parameters to substitute into the current state.

        Returns:
            Matrix of parameters with non-optimizable parameters appended
            in correct order.
        """
        self._check_substitutable(parameters)

        if isinstance(parameters, pd.DataFrame):
            parameters = parameters.reset_index(drop=True)  # Prevent index collisions.

        if isinstance(self.state, np.ndarray):
            return self._substitute_numpy(parameters)

        else:
            return self._substitute_pandas(parameters)

    def _check_substitutable(self, parameters: Matrix):
        """Check that the column indexing will work when we try to substitute.

        Args:
            parameters: Matrix of parameters to substitute into the current state.

        Raises:
            `TypeError`: if there is a type mismatch between the state and parameters.
            `ValueError`: if parameters has too many columns.
            `KeyError`: if the columns not present in the pandas case.
        """
        if len(self.optimizable_columns) != parameters.shape[-1]:
            raise ValueError(
                f"Parameter matrix has {parameters.shape[-1]} columns and we "
                f"have {len(self.optimizable_columns)} optimizable columns. "
                "Lengths must be the same."
            )

        if isinstance(parameters, pd.DataFrame):
            not_in_columns = [
                c for c in self.optimizable_columns if c not in parameters.columns
            ]
            if not_in_columns:
                raise KeyError(
                    f"Optimizable columns {not_in_columns} are not in the provided "
                    f"DataFrame with columns {parameters.columns}"
                )

    def _substitute_numpy(self, parameters: np.ndarray) -> np.ndarray:
        """Substitute the optimizable parameters into the current state for a numpy
        array.

        Args:
            parameters: numpy array of parameters to substitute into the current state.

        Returns:
            numpy array.
        """
        out = np.empty((parameters.shape[0], len(self.state)))

        out[:, self.optimizable_columns] = parameters
        out[:, self.non_optimizable_columns] = self.state[self.non_optimizable_columns]

        return out

    def _substitute_pandas(self, parameters: pd.DataFrame) -> pd.DataFrame:
        """Substitute the optimizable parameters into the current state for a numpy
        array.

        Args:
            parameters: numpy array of parameters to substitute into the current state.

        Returns:
            numpy array.
        """
        out = pd.DataFrame(parameters, columns=self.optimizable_columns)

        out[self.non_optimizable_columns] = self.state[
            self.non_optimizable_columns
        ].iloc[0]

        return out[self.state.columns]
