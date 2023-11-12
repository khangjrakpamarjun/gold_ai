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

import re
import typing as tp

import plotly.graph_objects as go
from matplotlib.colors import to_rgba as mpl_to_rgba
from plotly import colors

TDict = tp.Dict[str, tp.Any]


def get_default_fig_layout_params() -> tp.Tuple[TDict, TDict]:
    """Returns: default configurations for `fig_params` and `layout_params`"""

    default_fig_params = {}
    default_layout_params = dict(title=dict(x=0))  # noqa: C408
    return default_fig_params, default_layout_params


def apply_chart_style(
    fig: go.Figure,
    height: tp.Optional[int] = None,
    width: tp.Optional[int] = None,
    title: tp.Optional[str] = None,
    xaxis_title: tp.Optional[str] = None,
    yaxis_title: tp.Optional[str] = None,
    x: tp.Optional[str] = None,  # noqa: WPS111
    y: tp.Optional[str] = None,  # noqa: WPS111
    fig_params: tp.Optional[TDict] = None,
    layout_params: tp.Optional[TDict] = None,
    **kwargs: tp.Any,
) -> None:
    """
    Applies figure and layout parameters in the following order:
        * default params
        * `fig_params` and `layout_params`
        * all args and kwargs passed to this function

    Args:
        fig: Figure holding the chart traces
        height: the height of the exported image in layout pixels
        width: the width of the exported image in layout pixels
        title: Title of figure
        xaxis_title: x-axis title for the chart
        yaxis_title: y-axis title for the chart
        x: feature plotted on x-axis; used if xaxis_title is not passed
        y: feature plotted on y-axis; used if yaxis_title is not passed
        fig_params: additional parameters passed to configure traces
        layout_params: additional parameters passed to configure layout
        kwargs: layout arguments defined within function
    """

    default_fig_params, default_layout_params = get_default_fig_layout_params()
    fig.update_layout(default_layout_params)
    fig.update_traces(default_fig_params)

    fig.update_traces(fig_params)
    fig.update_layout(layout_params)

    properties_to_update = {
        "height": height,
        "width": width,
        "title": title,
        "xaxis_title": xaxis_title if xaxis_title is not None else x,
        "yaxis_title": yaxis_title if yaxis_title is not None else y,
        **kwargs,
    }
    properties_to_update_not_none = {
        property_name: property_value
        for property_name, property_value in properties_to_update.items()
        if property_value is not None
    }
    fig.update_layout(properties_to_update_not_none)


def add_watermark(
    fig: go.Figure,
    message: str,
    xref: tp.Optional[str] = None,
    yref: tp.Optional[str] = None,
) -> None:
    """
    Adds "watermark" annotation to the figure.

    Args:
        fig: Figure object to add annotation to
        message: Text for the annotation
        xref: annotation's x-axis
        yref: annotation's y-axis
    """
    annotation_size = 20
    fig.add_annotation(
        text=message,
        xref=xref,
        yref=yref,
        align="center",
        valign="middle",
        font=dict(size=annotation_size, color="grey"),  # noqa: C408
        showarrow=False,
    )


def cast_color_to_rgba(color: str, alpha: float = 1.0) -> str:
    """Converts color value in hex format to rgba format with alpha transparency

    Args:
        color: color in hex / rgba / text format.
            Examples: "Red", "rgba(255,25,125,0.7)", "#9267D3", "rgb(12,50,36)"
        alpha: opacity of the color

    Returns: rgba color
    """

    validated_color = colors.validate_colors([color])[0]
    validated_color = mpl_to_rgba(validated_color, alpha=alpha)
    return f"rgba{validated_color}"


def get_next_available_legend_group_name(
    fig: go.Figure,
    group_name: str,
    index_from: int = 0,
    id_separator: str = "_",
) -> str:
    escaped_group_name = re.escape(group_name)
    legend_naming_pattern = rf"{escaped_group_name}{id_separator}(\d)+$"
    legend_group_key = "legendgroup"

    def _trace_is_in_same_legend_group(trace):  # noqa: WPS430  # Clearer than a lambda
        """Checks if trace is in the legend group `group_name` (from the outer scope)"""
        return re.fullmatch(legend_naming_pattern, trace[legend_group_key] or "")

    similar_group_traces = fig.select_traces(_trace_is_in_same_legend_group)
    similar_group_suffixes = (
        int(re.sub(legend_naming_pattern, r"\1", trace[legend_group_key]))
        for trace in similar_group_traces
    )
    last_similar_group_suffix = max(similar_group_suffixes, default=index_from - 1)
    next_similar_group_suffix = last_similar_group_suffix + 1
    return f"{group_name}{id_separator}{next_similar_group_suffix}"
