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

import itertools
import logging
import typing as tp

import pandas as pd
import plotly.graph_objects as go

from reporting.charts.utils import (
    apply_chart_style,
    cast_color_to_rgba,
    check_data,
    get_error_range,
    get_next_available_legend_group_name,
)
from reporting.config import COLORS

logger = logging.getLogger(__name__)
_VIOLET = COLORS[0]


def draw_range(
    fig: go.Figure,
    data: pd.DataFrame,
    x: str,  # noqa: WPS111
    y: str,  # noqa: WPS111
    y_min: str,
    y_max: str,
    color: str = _VIOLET,
    range_opacity: float = 0.05,
    legend_name: tp.Optional[str] = None,
    upper_bound_name: str = "Upper Bound",
    lower_bound_name: str = "Lower Bound",
) -> None:
    """Create continuous error bar plot

    Args:
        data: dataframe for chart
        x: column name of column holding x axis data dataframe
        y: column name of column holding y axis data dataframe
        y_min: column name of column holding lower bound data for y axis
        y_max: column name of column holding upper bound data for y axis
        fig: plotly figure used for drawing plots
        color: sets the color of the line and error bar
        range_opacity: transparency of the color for error range
        legend_name: legend name
        upper_bound_name: description of the upper bound
        lower_bound_name: description of the lower bound

    Returns: plotly continuous error bar plot
    """

    check_data(data, x, y, y_min, y_max)
    legend_group_name = get_next_available_legend_group_name(fig, f"range_{y}")

    _draw_main_line(
        fig=fig,
        data=data,
        x=x,
        y=y,
        color=color,
        legend_name=legend_name,
        legend_group_name=legend_group_name,
    )

    _draw_confidence_interval(
        fig=fig,
        data=data,
        x=x,
        y_max=y_max,
        y_min=y_min,
        color=color,
        range_opacity=range_opacity,
        lower_bound_name=lower_bound_name,
        upper_bound_name=upper_bound_name,
        legend_group_name=legend_group_name,
    )


def _draw_main_line(
    fig: go.Figure,
    data: pd.DataFrame,
    x: str,  # noqa: WPS111
    y: str,  # noqa: WPS111
    color: str,
    legend_name: str,
    legend_group_name: str,
) -> None:
    fig.add_trace(
        go.Scatter(
            name=legend_name if legend_name is not None else y,
            x=data[x],
            y=data[y],
            mode="lines",
            legendgroup=legend_group_name,
            marker=dict(color=color),  # noqa: C408
            line=dict(width=3),  # noqa: C408
        ),
    )


def _draw_confidence_interval(
    fig: go.Figure,
    data: pd.DataFrame,
    x: str,  # noqa: WPS111
    y_max: str,
    y_min: str,
    color: str,
    range_opacity: float,
    lower_bound_name: str,
    upper_bound_name: str,
    legend_group_name: str,
) -> None:
    bound_params = dict(  # noqa: C408
        mode="lines",
        line=dict(width=0),  # noqa: C408
        marker=dict(color=color),  # noqa: C408
        showlegend=False,
        legendgroup=legend_group_name,
    )
    fig.add_trace(
        go.Scatter(name=upper_bound_name, x=data[x], y=data[y_max]).update(
            bound_params
        ),
    )
    fig.add_trace(
        go.Scatter(
            name=lower_bound_name,
            x=data[x],
            y=data[y_min],
            fillcolor=cast_color_to_rgba(color, range_opacity),
            fill="tonexty",
        ).update(bound_params),
    )


def draw_overlay(
    fig: go.Figure,
    data: pd.DataFrame,
    x: str,  # noqa: WPS111
    y: str,  # noqa: WPS111
    units: str,
    legendgroup: tp.Optional[str] = None,
    legendname: tp.Optional[str] = None,
    color: str = "Grey",
    opacity: float = 0.5,
) -> None:
    """Create overlay lines plot using groups from `units`

    Args:
        data: dataframe for chart
        x: column name of column holding x axis data dataframe
        y: column name of column holding y axis data dataframe
        units: column name in the data used for groupping. When used, a separate
                line will be drawn for each unit with appropriate semantics,
                but no legend entry will be added.
        legendgroup: name of the legend group
        legendname: name of the legend
        fig: plotly figure used for drawing plots
        color: sets the color of the line
        opacity: sets the transparency of the color for line

    Returns: plotly overlay chart
    """

    check_data(data, x, y, units)
    if legendgroup is None:
        legendgroup = get_next_available_legend_group_name(fig, "overlay")
    show_legend = True
    for _, unit_data in data.groupby(units):
        fig.add_trace(
            go.Scatter(
                x=unit_data[x],
                y=unit_data[y],
                legendgroup=legendgroup,
                name=legendname,
                mode="lines",
                line=dict(color=cast_color_to_rgba(color, opacity)),  # noqa: C408
                showlegend=show_legend,
            ),
        )
        show_legend = False


# TODO: If possible, refactor so the decision of what to plot (overlay vs range plot)
#     is taken at the beginning of the function, not inside the ``_draw_line_for_group``
# TODO: Reduce the cognitive complexity and remove the noqa WPS231 when doing the
#     refactoring suggested above.
# Ok to ignore WPS211 for plot functions with many plot options
def plot_lines(  # noqa: WPS211,WPS231
    data: pd.DataFrame,
    x: str,  # noqa: WPS111
    y: str,  # noqa: WPS111
    color: tp.Optional[str] = None,
    units: tp.Optional[str] = None,
    estimator: tp.Optional[str] = None,
    error_method: str = "ci",
    error_level: int = 95,
    opacity: float = 0.5,
    fig: tp.Optional[go.Figure] = None,
    title: str = "Line Plot",
    color_map: tp.Optional[tp.Dict[str, str]] = None,
    xaxis_title: tp.Optional[str] = None,
    yaxis_title: tp.Optional[str] = None,
    height: tp.Optional[int] = None,
    width: tp.Optional[int] = None,
    fig_params: tp.Optional[tp.Dict[str, tp.Any]] = None,
    layout_params: tp.Optional[tp.Dict[str, tp.Any]] = None,
) -> go.Figure:
    """Create statistical line plot. Plotly equivalent to seaborn line plot.

    Args:
        data: : dataframe for chart
        x: column name of column holding x axis data dataframe
        y: column name of column holding y axis data dataframe
        color: name of the grouping variable that will produce lines
                with different colors
        units: grouping variable identifying line units. When used, a separate
                line will be drawn for each unit but no legend entry will be added
        estimator: sets the method for aggregating across multiple observations of
                    the y variable at the same x level. If None, all observations
                    will be drawn
        error_method: sets the method to determine the size of the confidence interval
                      to draw when aggregating with an estimator
        error_level: level of the confidence interval
        opacity: sets the transparency of the color
        fig: plotly figure used for drawing plots
        title: title for the chart
        color_map: mapping from `color` column groups to line color
        xaxis_title: x-axis title for the chart
        yaxis_title: y-axis title for the chart
        height: the height of the exported image in layout pixels
        width: the width of the exported image in layout pixels
        fig_params: kwargs for plotly chart function
        layout_params: dictionary containing keys and values for updating the plotly
         fig layout

    Raises:
        ValueError: raises error when estimator and units have values at the same time

    Returns: plotly line plot with possibility of several statistical groupings.
    """

    check_data(data, x, y)

    is_range_plot = estimator is not None
    is_overlay_plot = units is not None
    if is_range_plot and is_overlay_plot:
        raise ValueError("Pass either `estimator` or `units` argument, not both.")

    if fig is None:
        fig = go.Figure()

    if color is None:
        if color_map is not None:
            logger.warning(
                "Arg `color_map` is used only in case `color` arg is provided. "
                "Will be ignored in this case.",
            )
        fig = _draw_line_for_group(
            fig=fig,
            group_data=data,
            x=x,
            y=y,
            units=units,
            estimator=estimator,
            error_method=error_method,
            error_level=error_level,
            group_name=None,
            group_color=COLORS[0],
            opacity=opacity,
        )
    else:
        data_by_color = data.groupby(color)
        colors_cycle = itertools.cycle(COLORS)
        for (group_name, group_data), group_color in zip(data_by_color, colors_cycle):
            fig = _draw_line_for_group(
                fig=fig,
                group_data=group_data,
                x=x,
                y=y,
                units=units,
                estimator=estimator,
                error_method=error_method,
                error_level=error_level,
                group_name=group_name,
                group_color=(
                    group_color
                    if color_map is None
                    else color_map.get(group_name, group_color)
                ),
                opacity=opacity,
            )
    apply_chart_style(
        fig=fig,
        height=height,
        width=width,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        x=x,
        y=y,
        fig_params=fig_params,
        layout_params=layout_params,
    )
    if is_range_plot:
        fig.update_layout(
            title=title if title else f"Range Plot for Sensor {y}",
            hovermode="x",
        )
    elif is_overlay_plot:
        fig.update_layout(
            title=title if title else f"Overlay Plot for Sensor {y}",
            legend_title_text=color,
        )
    return fig


def _draw_line_for_group(
    fig: go.Figure,
    group_data: pd.DataFrame,
    x: str,  # noqa: WPS111
    y: str,  # noqa: WPS111
    units: tp.Optional[str],
    estimator: tp.Optional[str],
    error_method: str,
    error_level: int,
    group_name: tp.Optional[str],
    group_color: str,
    opacity: float,
) -> go.Figure:
    """Draws either range or overlay plot based on what is passed - estimator or units"""

    if estimator is not None:
        group_data = (
            group_data.groupby(x)
            .apply(
                get_error_range,
                variable_name=y,
                estimator=estimator,
                error_method=error_method,
                error_level=error_level,
            )
            .reset_index()
        )
        draw_range(
            fig=fig,
            data=group_data,
            x=x,
            y=y,
            y_min=f"{y}_min",
            y_max=f"{y}_max",
            color=group_color,
            range_opacity=opacity,
            legend_name=group_name,
            upper_bound_name="CI Upper Bound",
            lower_bound_name="CI Lower Bound",
        )
    elif units is not None:
        draw_overlay(
            fig=fig,
            data=group_data,
            x=x,
            y=y,
            units=units,
            legendgroup=group_name,
            legendname=group_name,
            color=group_color,
            opacity=opacity,
        )
    return fig
