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
from plotly import graph_objects as go
from plotly.express.colors import qualitative as colors
from plotly.graph_objs import Figure as PlotlyFigure

from optimizer import Repair, StatefulOptimizationProblem, UserDefinedRepair
from optimizer.types import Matrix, Vector

from .layout import calculate_y_range_limits

REPAIR_OPACITY = 0.2
SCATTER_MARKER_SIZE_INITIAL_POSITION = 10
SCATTER_MARKER_SIZE_OPTIMIZED_POSITION = 10
VERTICAL_LINE_ANNOTATION_TEXT_ANGLE = 0
VERTICAL_LINE_FONT_SIZE = 12
VERTICAL_LINE_OPACITY = 0.75


def create_line_trace(
    dependent_function: tp.Callable[[Matrix], Vector],
    optimizable_parameter: str,
    state_with_grid: pd.DataFrame,
) -> go.Scatter:
    """
    Create line+markers scatter plot of the
    ``dependent_function(optimizable_parameter)`` vs ``optimizable_parameter``
    """
    return go.Scatter(
        x=state_with_grid[optimizable_parameter],
        y=dependent_function(state_with_grid),
        mode="lines+markers",
        showlegend=False,
        marker={
            "color": "black",
            "size": 5,
            "line": {
                "color": "black",
                "width": 1,
            },
        },
    )


def create_initial_position_scatter_trace(
    initial_position: float,
    initial_dependent_function_value: float,
) -> go.Scatter:
    """
    Create single points scatter trace representing
    the initial state of the ``optimizable_parameter``.
    """
    return go.Scatter(
        x=[initial_position],
        y=[initial_dependent_function_value],
        fillcolor="black",
        mode="markers",
        marker={"size": SCATTER_MARKER_SIZE_OPTIMIZED_POSITION, "color": "black"},
        showlegend=False,
    )


def create_optimized_position_scatter_trace(
    optimized_position: float,
    optimized_dependent_function_value: float,
) -> go.Scatter:
    """
    Create single points scatter trace representing
    the optimized state of the ``optimizable_parameter``.
    """
    return go.Scatter(
        x=[optimized_position],
        y=[optimized_dependent_function_value],
        fillcolor="black",
        mode="markers",
        marker={"size": SCATTER_MARKER_SIZE_OPTIMIZED_POSITION, "color": "black"},
        showlegend=False,
    )


def _create_repair_violations_to_plot(
    problem: StatefulOptimizationProblem,
    state_with_grid: pd.DataFrame,
) -> tp.Tuple[tp.List[Repair], tp.List[np.ndarray]]:
    """
    Return user defined repairs and points of the grid where they are violated.
    """
    user_defined_repairs = [
        repair for repair in problem.repairs if isinstance(repair, UserDefinedRepair)
    ]
    repair_violations = []
    for repair in user_defined_repairs:
        repair(state_with_grid)
        repair_violations.append(repair.constraint.violated)
    return user_defined_repairs, repair_violations


def create_repairs_scatter_traces(
    problem: StatefulOptimizationProblem,
    dependent_function: tp.Callable[[Matrix], Vector],
    grid: tp.List[float],
    state_with_grid: pd.DataFrame,
    initial_dependent_function_value: float,
) -> tp.List[go.Scatter]:
    """
    Create scatter traces that are filled inside
    that highlight regions constrained with ``UserDefinedRepairs``.

    Notes:
        All regions related to one repair have
        same color and item in legend of the plot.
    """
    repairs_to_plot, violations = _create_repair_violations_to_plot(
        problem=problem,
        state_with_grid=state_with_grid,
    )
    scatters = []
    for repair, violation, color in zip(repairs_to_plot, violations, colors.Set1):
        violation = list(violation)
        # All regions related to one repair have same color and item in legend
        # This is done through manipulating ``showlegend`` kw argument.
        # See "Grouped Legend Items" section at plotly.com/python/legend/
        # for more details.
        showlegend = True
        for grid_index in range(len(grid)):  # noqa: WPS518
            if violation[grid_index]:
                scatters.append(
                    _create_single_repair_scatter_trace(
                        dependent_function(state_with_grid),
                        initial_dependent_function_value,
                        grid_index=grid_index,
                        parameter_grid=grid,
                        color=color,
                        scatter_group_name=repair.name,
                        showlegend=showlegend,
                    ),
                )
                showlegend = False
    return scatters


def _create_single_repair_scatter_trace(
    dependent_function_values: Vector,
    initial_dependent_function_value: float,
    grid_index: int,
    parameter_grid: tp.List[float],
    color: str,
    scatter_group_name: str,
    showlegend: bool,
) -> go.Scatter:
    if grid_index == 0:
        lhs_coordinate = parameter_grid[0]
    else:
        lhs_coordinate = (
            parameter_grid[grid_index]
            - (parameter_grid[grid_index] - parameter_grid[grid_index - 1]) / 2
        )
    if grid_index == len(parameter_grid) - 1:
        rhs_coordinate = parameter_grid[-1]
    else:
        rhs_coordinate = (
            parameter_grid[grid_index]
            + (parameter_grid[grid_index + 1] - parameter_grid[grid_index]) / 2
        )
    lower_coordinate, upper_coordinate = calculate_y_range_limits(
        list(dependent_function_values) + [initial_dependent_function_value],
    )
    return go.Scatter(
        x=[
            lhs_coordinate,
            lhs_coordinate,
            rhs_coordinate,
            rhs_coordinate,
            lhs_coordinate,
        ],
        y=[
            lower_coordinate,
            upper_coordinate,
            upper_coordinate,
            lower_coordinate,
            lower_coordinate,
        ],
        mode="lines",
        line={"color": "rgba(0,0,0,0)"},
        fill="toself",
        legendgroup=scatter_group_name,
        name=scatter_group_name,
        fillcolor=color,
        showlegend=showlegend,
        opacity=REPAIR_OPACITY,
    )


def add_vertical_line(
    fig: go.Figure,
    x_axis_coordinate: float,
    y_axis_coordinate: float,
    annotation_position: str,
    text: str,
) -> None:
    y_min, y_max = tuple(fig.layout.yaxis.range)
    fig.add_shape(
        go.layout.Shape(
            type="line",
            x0=x_axis_coordinate,
            x1=x_axis_coordinate,
            y0=y_axis_coordinate,
            y1=y_max,
            line={"color": "Black", "width": 1},
            opacity=VERTICAL_LINE_OPACITY,
        ),
    )
    fig.add_annotation(
        go.layout.Annotation(
            showarrow=True,
            text=text,
            x=x_axis_coordinate,
            xanchor="left",
            y=y_max,
            yanchor=annotation_position,
            textangle=VERTICAL_LINE_ANNOTATION_TEXT_ANGLE,
            font={"size": VERTICAL_LINE_FONT_SIZE},
            opacity=VERTICAL_LINE_OPACITY,
        ),
    )


def add_optimization_limits(
    fig: PlotlyFigure,
    lhs_bound: float,
    rhs_bound: float,
) -> None:
    fig.add_vline(x=lhs_bound, line_dash="dash", line_color="red")
    fig.add_vline(x=rhs_bound, line_dash="dash", line_color="red")
