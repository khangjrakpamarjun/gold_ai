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


import typing as tp

import numpy as np
import pandas as pd

from optimizer import StatefulOptimizationProblem
from optimizer.solvers.continuous.base import ContinuousSolver
from optimizer.types import Matrix, Vector


class ObjectiveWithPenalties(object):
    def __init__(self, problem: StatefulOptimizationProblem):
        self._problem = problem

    def __call__(self, input_parameters: Matrix) -> Vector:
        objective_value = self._problem.objective(input_parameters)
        if self._problem.penalties:
            objective_value += self._problem.apply_penalties(input_parameters)
        return objective_value

    def __str__(self):
        return "Objective with penalties"


def extract_objective_with_penalties(
    problem: StatefulOptimizationProblem,
) -> tp.Callable[[Matrix], Vector]:
    return ObjectiveWithPenalties(problem)


def extract_optimized_state(
    problem: StatefulOptimizationProblem,
    solver: ContinuousSolver,
) -> pd.DataFrame:
    """
    Extracts optimized state from the problem and solver
    """
    state = problem.state.copy()
    best_parameters, _ = solver.best()
    state[problem.optimizable_columns] = best_parameters
    return state


def extract_bounds_for_parameter(
    optimizable_parameter: str,
    problem: StatefulOptimizationProblem,
    solver: ContinuousSolver,
) -> tp.Tuple[float, float]:
    """
    Extracts optimization bounds from solver domain
    """
    parameter_index = problem.optimizable_columns.index(optimizable_parameter)
    # TODO: return ``solver.domain[parameter_index]``
    # TODO: once issue with optimization bounds in resolved (#2666)
    bounds = solver.domain[parameter_index]
    return min(bounds), max(bounds)


def extract_initial_value_for_parameter(
    optimizable_parameter: str,
    problem: StatefulOptimizationProblem,
) -> float:
    """
    Extract unoptimized (initial) value for parameter
    """
    return problem.state[optimizable_parameter].iloc[0]


def extract_optimized_value_for_parameter(
    optimizable_parameter: str,
    problem: StatefulOptimizationProblem,
    solver: ContinuousSolver,
) -> float:
    """
    Extract optimized value for parameter
    """
    parameter_index = problem.optimizable_columns.index(optimizable_parameter)
    best_parameters, _ = solver.best()
    return best_parameters[parameter_index]


def extract_dependent_function_value(
    dependent_function: tp.Callable[[Matrix], Vector],
    state: pd.DataFrame,
    optimizable_parameter_value: float,
    optimizable_parameter: str,
) -> float:
    """
    Returns the value of the dependent function as a single float number.

    Handles the cases where the dependent function
    returns a ``pd.Series`` or a ``np.ndarray``, raises an error otherwise
    """
    state = state.copy()
    state[optimizable_parameter] = optimizable_parameter_value
    dependent_function_value = dependent_function(state)
    if isinstance(dependent_function_value, pd.Series):
        return dependent_function_value.iloc[0]
    elif isinstance(dependent_function_value, np.ndarray):
        return dependent_function_value[0]
    unexpected_dependent_function_return_type = type(dependent_function_value)
    raise ValueError(
        "Unexpected return type for `dependent_function`:"
        f" {unexpected_dependent_function_return_type}",
    )
