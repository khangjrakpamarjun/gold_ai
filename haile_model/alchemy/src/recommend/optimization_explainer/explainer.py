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

import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objects import Figure as PlotlyFigure

from optimizer.problem import StatefulOptimizationProblem
from optimizer.solvers.continuous.base import ContinuousSolver
from optimizer.types import Matrix, Vector

from .extraction import (
    extract_bounds_for_parameter,
    extract_dependent_function_value,
    extract_initial_value_for_parameter,
    extract_objective_with_penalties,
    extract_optimized_state,
    extract_optimized_value_for_parameter,
)
from .grid import create_grid_for_optimizable_parameter
from .layout import create_layout_for_optimization_explainer
from .trace import (
    add_optimization_limits,
    add_vertical_line,
    create_initial_position_scatter_trace,
    create_line_trace,
    create_optimized_position_scatter_trace,
    create_repairs_scatter_traces,
)


def create_optimization_explainer_plot(  # noqa: WPS210
    optimizable_parameter: str,
    problem: StatefulOptimizationProblem,
    solver: ContinuousSolver,
    dependent_function: tp.Optional[tp.Callable[[Matrix], Vector]] = None,
    state: tp.Optional[pd.DataFrame] = None,
    n_points_in_grid: tp.Optional[int] = None,
    x_axis_name: tp.Optional[str] = None,
    y_axis_name: tp.Optional[str] = None,
) -> PlotlyFigure:
    """
    Create plot that explains why optimized
    value is better than initial value of the parameter.

    Shows the dependencies between different values
    of ``optimizable_parameter`` and ``dependent_function(optimizable_parameter)``.
    Highlights areas constrained with repairs
    and marks initial and optimized values for ``optimizable_parameter``.

    Args:
        dependent_function: Function used for Y-axis that maps
         row with parameters into number.
         Can be optimization objective, penalty,
         or anything else aligned with the API.
        optimizable_parameter: Optimization parameter name involved in optimization
        problem: problem involved in optimization routine
        solver: Continuous solver involved in optimization routine.
        state: Values used for other parameters needed
         to calculate ``dependent_function(optimizable_parameter)``
        n_points_in_grid: number of points to use
         in grid if grid can not be produced from problem
        x_axis_name: Name for X-axis to use. Might be useful
         when parameters involved into optimization don't have human-readable name
        y_axis_name: Name for X-axis to use. By 'dependent_function' class name
         is used by default. Hence, this parameter is useful,
         when default class name is not intuitive
    Returns:
        plotly.Figure with optimization explanation
    """
    if state is None:
        state = extract_optimized_state(problem, solver)
    if dependent_function is None:
        dependent_function = extract_objective_with_penalties(problem)
    lhs_bound, rhs_bound = extract_bounds_for_parameter(
        optimizable_parameter=optimizable_parameter,
        problem=problem,
        solver=solver,
    )
    initial_parameter_position = extract_initial_value_for_parameter(
        optimizable_parameter,
        problem,
    )
    initial_dependent_function_value = extract_dependent_function_value(
        dependent_function=dependent_function,
        state=state,
        optimizable_parameter_value=initial_parameter_position,
        optimizable_parameter=optimizable_parameter,
    )
    optimized_parameter_position = extract_optimized_value_for_parameter(
        optimizable_parameter=optimizable_parameter,
        problem=problem,
        solver=solver,
    )
    optimized_dependent_function_value = extract_dependent_function_value(
        dependent_function=dependent_function,
        state=state,
        optimizable_parameter_value=optimized_parameter_position,
        optimizable_parameter=optimizable_parameter,
    )
    parameter_grid = create_grid_for_optimizable_parameter(
        problem=problem,
        optimizable_parameter=optimizable_parameter,
        lhs_bound=lhs_bound,
        rhs_bound=rhs_bound,
        initial_position=initial_parameter_position,
        optimized_position=optimized_parameter_position,
        n_points_in_grid=n_points_in_grid,
    )
    state_with_grid = _fill_state_with_grid(
        parameter_grid=parameter_grid,
        state=state,
        optimizable_parameter=optimizable_parameter,
    )
    fig = go.Figure(
        data=[
            *create_repairs_scatter_traces(
                problem=problem,
                dependent_function=dependent_function,
                grid=parameter_grid,
                state_with_grid=state_with_grid,
                initial_dependent_function_value=initial_dependent_function_value,
            ),
            create_line_trace(
                dependent_function=dependent_function,
                optimizable_parameter=optimizable_parameter,
                state_with_grid=state_with_grid,
            ),
            create_initial_position_scatter_trace(
                initial_position=initial_parameter_position,
                initial_dependent_function_value=initial_dependent_function_value,
            ),
            create_optimized_position_scatter_trace(
                optimized_position=optimized_parameter_position,
                optimized_dependent_function_value=optimized_dependent_function_value,
            ),
        ],
        layout=create_layout_for_optimization_explainer(
            optimizable_parameter,
            dependent_function,
            state_with_grid,
            initial_dependent_function_value,
            x_axis_name,
            y_axis_name,
        ),
    )
    add_vertical_line(
        fig,
        x_axis_coordinate=initial_parameter_position,
        y_axis_coordinate=initial_dependent_function_value,
        text=f"Initial: {initial_parameter_position:0.2f}",
        annotation_position="top",
    )
    add_vertical_line(
        fig,
        x_axis_coordinate=optimized_parameter_position,
        y_axis_coordinate=optimized_dependent_function_value,
        text=f"Optimized: {optimized_parameter_position:0.2f}",
        annotation_position="bottom",
    )
    add_optimization_limits(fig, lhs_bound=lhs_bound, rhs_bound=rhs_bound)
    return fig


def _fill_state_with_grid(
    parameter_grid: tp.List[float],
    state: pd.DataFrame,
    optimizable_parameter: str,
) -> pd.DataFrame:
    state_with_grid = pd.concat((state for _ in range(len(parameter_grid))))
    state_with_grid[optimizable_parameter] = parameter_grid
    return state_with_grid
