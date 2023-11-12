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
Holds definitions for the StatefulContextualOptimizationProblem
"""

from copy import deepcopy
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from optimizer.constraint.penalty import Penalty
from optimizer.constraint.repair import Repair
from optimizer.problem.problem import OptimizationProblem
from optimizer.types import Matrix, Vector
from optimizer.utils.functional import safe_subset
from optimizer.utils.validation import check_matrix


class StatefulContextualOptimizationProblem(OptimizationProblem):
    """
    Optimization problem representing the case where both optimizable and
    non-optimizable parameters must be handled.
    """

    def __init__(
        self,
        objective: Callable,
        state: Vector,
        context_data: Union[pd.DataFrame, np.ndarray],
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
        objective_kwargs: Dict = None,
        n_opt_blocks: int = 1,
    ):
        """Create a new instance of a Stateful Contextual Problem..

        Args:
            objective: callable object representing the objective function.
            state: Vector representing the current state of the problem.
            optimizable_columns: list of columns/dimensions that can be optimized.
            penalties: optional Penalty or list of Penalties.
            repairs: optional Repairs or callable or list of Repairs and/or callables.
            sense: 'minimize' or 'maximize', how to optimize the objective.
            n_opt_blocks: int (1 by default). How many timesteps to optimize
            over (constraining to be the same value). When value is 1, only
             most recent timestep is optimized.
        """
        super(StatefulContextualOptimizationProblem, self).__init__(
            objective, penalties=penalties, repairs=repairs, sense=sense
        )
        # An internal current state will save us time for any processing
        # we might need to apply to the current state before subbing in
        # the optimizable variables in the __call__ method.
        self.optimizable_columns = optimizable_columns
        self.context_data = context_data
        self.state = state
        self.n_opt_blocks = n_opt_blocks
        self.obj_kwargs = objective_kwargs

    @property
    def objective(self) -> Callable[[Matrix], Vector]:
        """Returns callable objective function."""
        if callable(self._objective):
            return self._objective

        def _predict(df, tiled=False):
            """
            Predict on values relevant for a single time period.

            Args:
                df:
                tiled: boolean, controls if each row of `df` should be
                predicted with the same context data.

            Returns:
                A dataframe of the same dimensionality as df,
                provided predict returns a single value for a time-series of
                values.

            """
            if not tiled:
                self._update_internal_state(df)
                df = self.substitute_parameters(df[self.optimizable_columns])
                # Now return predictions on self._internal_state
            return self._objective.predict(df, **self.obj_kwargs)

        return _predict

    @property
    def optimizable_columns(self) -> Union[List, pd.Series]:
        """Optimizable columns getter.

        Returns:
            List or pd.Series.
        """
        return self._optimizable_columns

    @optimizable_columns.setter
    def optimizable_columns(self, new_columns: Union[List, pd.Series]):
        """Optimizable columns setter.
        Resets internal state.

        Args:
            new_columns: list of columns/dimensions that can be optimized.
        """
        self._optimizable_columns = new_columns
        self._internal_state = None  # Reset internal state.

    @property
    def state(self) -> Vector:
        """
        Returns the current (actual) state

        Returns:
            vector representing the current state of the problem
        """
        return self._actual_state

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
        difference = set(self._optimizable_columns) - set(all_columns)
        if difference:
            raise KeyError(
                f"Provided optimizable columns not found in provided state: "
                f"{sorted(list(difference))}"
            )
        self._internal_state = None
        self._actual_state = pd.concat([self._context_data, new_state], axis=0)

    @property
    def context_data(self) -> Union[np.ndarray, pd.DataFrame]:
        """Context Data getter.

        Returns:
            2d data array
        """
        return self._context_data

    @context_data.setter
    def context_data(self, new_data: Union[pd.DataFrame, np.ndarray]):
        """Optimizable columns setter.
        Resets internal state.

        Args:
            Context Data
        """
        self._context_data = new_data

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
        if self.repairs and not skip_repairs:
            params_plus_state = self.apply_repairs(params_plus_state)
        objectives = self.objective(params_plus_state, tiled=True)
        if self.penalties and not skip_penalties:
            objectives += self.apply_penalties(params_plus_state)
        start = self._context_data.shape[0]
        stride = start + 1
        # N.B.->Firsts entry of objectives starts to prediction at
        # parameters.iloc[start]
        return (
            objectives,
            safe_subset(
                params_plus_state.iloc[start::stride, :], self._optimizable_columns
            ),
        )

    def substitute_parameters(self, parameters: Matrix) -> Matrix:
        """Substitute the optimizable parameters into the current state.

        Args:
            parameters: Matrix of parameters to substitute into the current state.

        Returns:
            Matrix of parameters with non-optimizable parameters appended
            in correct order.
        """
        self._check_shape(parameters)
        # N context rows end at index N-1.
        start = self._context_data.shape[0]
        stride = start + 1
        # Removes possible collisions with indexes.
        if isinstance(parameters, pd.DataFrame):
            parameters = parameters.values

        if isinstance(self._actual_state, np.ndarray):
            for i in range(self.n_opt_blocks):
                self._internal_state[
                    max(start - i, 0) :: stride, self._optimizable_columns
                ] = parameters

        else:
            for i in range(self.n_opt_blocks):
                self._internal_state.loc[
                    max(start - i, 0) :: stride, self._optimizable_columns
                ] = parameters

        return self._internal_state

    def _check_shape(self, parameters: Matrix):
        """Check the shape of the parameters matrix.
        Update the internal params_shape and internal current state if
        the shape of the parameters matrix has changed.

        Args:
            parameters: Matrix of parameters to evaluate.
        """
        if (
            self._internal_state is None
            or self._internal_state.shape[0] != parameters.shape[0]
        ):
            self._update_internal_state(parameters)

    def _update_internal_state(self, parameters):
        """Update the internal current state variable.

        Args:
            parameters: Matrix of parameters to use to update the internal state.
        """

        n_tile = parameters.shape[0]
        if isinstance(self._actual_state, np.ndarray):
            self._internal_state = np.tile(self._actual_state, (n_tile, 1))

        else:
            self._internal_state = pd.concat(
                [self._actual_state] * n_tile, ignore_index=True
            )
