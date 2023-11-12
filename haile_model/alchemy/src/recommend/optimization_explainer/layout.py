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

from optimizer.constraint.handler import BaseHandler
from optimizer.types import Matrix, Vector

EPSILON_FOR_BORDERS = 1e-4
EXTRA_RANGE_FRACTION = 0.1


def create_layout_for_optimization_explainer(
    optimizable_parameter: str,
    dependent_function: tp.Callable[[Matrix], Vector],
    state_with_grid: pd.DataFrame,
    initial_dependent_function_value: float,
    x_axis_name: tp.Optional[str] = None,
    y_axis_name: tp.Optional[str] = None,
) -> go.Layout:
    dependent_function_values = list(dependent_function(state_with_grid)) + [
        initial_dependent_function_value
    ]
    x_axis_name = x_axis_name if x_axis_name is not None else optimizable_parameter
    y_axis_name = (
        y_axis_name
        if y_axis_name is not None
        else _extract_string_representation(dependent_function)
    )
    return go.Layout(
        xaxis={"title": x_axis_name},
        yaxis={
            "title": y_axis_name,
            "range": calculate_y_range_limits(dependent_function_values),
        },
    )


def calculate_y_range_limits(
    dependent_function_values: tp.Iterable[float],
) -> tp.Tuple[float, float]:
    dependent_function_values = list(dependent_function_values)
    dependent_function_min = min(dependent_function_values)
    dependent_function_max = max(dependent_function_values)
    y_range_lower_limit = (
        dependent_function_min
        - (dependent_function_max - dependent_function_min) * EXTRA_RANGE_FRACTION
        - EPSILON_FOR_BORDERS
    )
    y_range_upper_limit = (
        dependent_function_max
        + (dependent_function_max - dependent_function_min) * EXTRA_RANGE_FRACTION
        + EPSILON_FOR_BORDERS
    )
    return y_range_lower_limit, y_range_upper_limit


def _extract_string_representation(
    dependent_function: tp.Callable[[Matrix], Vector],
) -> str:
    if isinstance(dependent_function, BaseHandler):
        return dependent_function.name
    return str(dependent_function)
