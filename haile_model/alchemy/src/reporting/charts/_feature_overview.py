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

"""Contains feature overview plot `plot_feature_overviews`"""

import textwrap
import typing as tp

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from reporting.charts.utils import add_watermark, apply_chart_style, check_data
from reporting.config import COLORS

_DEFAULT_COLOR = COLORS[0]
_DEFAULT_TABLE_FONT_SIZE = 11
_DEFAULT_GRAPH_FONT_SIZE = 11
_DEFAULT_ANNOTATION_TEXT_ANGLE = 90
_DEFAULT_X_TITLE_NORMALIZED_POSITION = 0.15
_DEFAULT_FIGURE_HEIGHT = 700

TDict = tp.Dict[str, tp.Any]
TRange = tp.Tuple[float, float]
TRangeOrDictWithRange = tp.Union[TRange, tp.Dict[str, TRange]]


def _generate_stats(
    data: pd.DataFrame,
    feature: str,
    precision: int = 4,
) -> pd.DataFrame:
    """Generate descriptive statistics of a feature

    Args:
        data: dataframe that contains the feature data
        feature: name of the feature
        precision: number of decimal places to round each column to

    Returns: summary statistics of the feature provided.
    """

    check_data(data, feature)
    return data[feature].describe().round(precision).reset_index(level=0).T


def _add_range(
    fig: go.Figure,
    row: int,
    col: int,
    tag_range: TRange,
    direction: str = "h",
    **kwargs: tp.Any,
) -> None:
    """This function adds range_min/max to plotly figure.

    Args:
        fig: plotly graph figure object
        row: row of subplot to add range to
        col: col of subplot to add range to
        tag_range: minimum and maximum value to show on the plot
        direction: direction of lines. Can only be "h" or "v". Defaults to "h".
        **kwargs: all kwargs are passed to `add_hline` / `add_vline`

    Raises:
        ValueError: raises when direction is neither "h" nor "v"
    """
    range_min, range_max = tag_range

    if direction not in {"h", "v"}:
        raise ValueError("direction must be either 'h' or 'v'")

    line_config = dict(line_width=1, line_color="red", line_dash="dash")  # noqa: C408

    if not np.isnan(range_min) and not np.isinf(range_min):
        annotation_text = "min"
        if direction == "h":
            fig.add_hline(
                y=range_min,
                row=row,
                col=col,
                annotation_text=annotation_text,
                annotation_position="top right",
                **line_config,
                **kwargs,
            )
        else:
            fig.add_vline(
                x=range_min,
                row=row,
                col=col,
                annotation_text=annotation_text,
                annotation_textangle=_DEFAULT_ANNOTATION_TEXT_ANGLE,
                **line_config,
                **kwargs,
            )

    if not np.isnan(range_max) and not np.isinf(range_max):
        annotation_text = "max"
        if direction == "h":
            fig.add_hline(
                y=range_max,
                row=row,
                col=col,
                annotation_text=annotation_text,
                **line_config,
                annotation_position="bottom left",
                **kwargs,
            )
        else:
            fig.add_vline(
                x=range_max,
                row=row,
                col=col,
                annotation_text=annotation_text,
                **line_config,
                annotation_position="top left",
                annotation_textangle=_DEFAULT_ANNOTATION_TEXT_ANGLE,
                **kwargs,
            )


def _add_overview_first_row_plots(
    fig: go.Figure,
    feature: pd.Series,
    feature_compact_name: str,
    legend_group: str,
    target: tp.Optional[pd.Series] = None,
) -> None:
    """Makes the plots for the first row of the overview plot.

    Can modifies ``fig`` in place.

    ``feature`` and ``target`` are the series (``data[feature]`` and ``data[target]``),
    not their names.

    These are:
    - box plot of the feature values
    - histogram of the feature value
    - scatterplot to explore the relationship between feature and target
        - replace by a watermark saying that target was not provided if this is the case
    """
    fig.add_trace(
        go.Box(
            y=feature,
            name=feature_compact_name,
            marker_color=_DEFAULT_COLOR,
            legendgroup=legend_group,
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    # todo: use nice distplot
    fig.add_trace(
        go.Histogram(
            x=feature,
            name=feature_compact_name,
            marker_color=_DEFAULT_COLOR,
            legendgroup=legend_group,
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    if target is not None:
        fig.add_trace(
            go.Scattergl(
                x=feature,
                y=target,
                name=feature_compact_name,
                mode="markers",
                marker_color=_DEFAULT_COLOR,
                legendgroup=legend_group,
            ),
            row=1,
            col=3,
        )
    else:
        add_watermark(fig, message="No target provided", xref="x3", yref="y3")


def _add_overview_second_row_table(
    fig: go.Figure,
    stats: pd.DataFrame,
) -> None:
    """Adds the table to the second row

    Modifies ``fig`` in place.

    This table is a table with a summary of the stats about the feature
    """

    # todo: update used table in the view
    fig.add_trace(
        go.Table(
            header=dict(  # noqa: C408
                values=["<b>statistic</b>", "<b>value</b>"],
                fill_color="LightGray",
                line_color="black",
                align="left",
                font=dict(color="black", size=_DEFAULT_TABLE_FONT_SIZE),  # noqa: C408
            ),
            cells=dict(  # noqa: C408
                values=stats,
                fill_color="white",
                line_color="black",
                align="left",
                font=dict(color="black", size=_DEFAULT_TABLE_FONT_SIZE),  # noqa: C408
            ),
        ),
        row=2,
        col=3,
    )


def _add_overview_second_row_plot(
    fig: go.Figure,
    feature: pd.Series,
    timestamp: pd.Series,
    feature_compact_name: str,
    legend_group: str,
) -> None:
    """Makes the plot for the second row of the overview plot and adds it to ``fig``.

    ``feature`` and ``timestamp`` are the series (``data[feature]`` and
    ``data[timestamp]``), not their names.

    This plot is a scatter plot of the timeseries of the feature values
    """
    fig.add_trace(
        go.Scatter(
            x=timestamp,
            y=feature,
            name=feature_compact_name,
            mode="lines",
            marker_color=_DEFAULT_COLOR,
            legendgroup=legend_group,
            showlegend=False,
        ),
        row=2,
        col=1,
    )


def _update_overview_layout(
    fig: go.Figure,
    title: str,
    feature: str,
    feature_compact_name: str,
    target_compact_name: tp.Optional[str],
):
    """Updates the layout of the feature overview figure.

    Modifies ``fig`` in place.
    """
    fig.update_layout(
        title=title if title else f"{feature} - Overview",
        title_x=_DEFAULT_X_TITLE_NORMALIZED_POSITION,
        template="plotly",
        showlegend=True,
        height=_DEFAULT_FIGURE_HEIGHT,
        xaxis=dict(title=None, showticklabels=False),  # noqa: C408
        yaxis=dict(title=feature_compact_name),  # noqa: C408
        yaxis2=dict(title="Frequency"),  # noqa: C408
        xaxis2=dict(title=feature_compact_name),  # noqa: C408
        yaxis3=dict(title=target_compact_name),  # noqa: C408
        xaxis3=dict(title=feature_compact_name),  # noqa: C408
        yaxis4=dict(title=feature_compact_name),  # noqa: C408
        font=dict(size=_DEFAULT_GRAPH_FONT_SIZE),  # noqa: C408
    )
    fig.update_xaxes(showgrid=True)  # triggers recursive update
    fig.update_yaxes(showgrid=True)  # triggers recursive update


def _plot_feature_overview(
    data: pd.DataFrame,
    feature: str,
    timestamp: str,
    target: tp.Optional[str] = None,
    tag_range: tp.Optional[TRange] = None,
    title: tp.Optional[str] = None,
    labels_length_limit: int = 20,
    fig_params: tp.Optional[TDict] = None,
    layout_params: tp.Optional[TDict] = None,
) -> go.Figure:
    """
    Create plots a collection of plots representing chosen feature "overview".
    This includes a boxplot and histogram to understand the distribution of values,
    a scatter-plot vs the `target` variable, and a time-series plot of the `feature` and
    `target`.

    Args:
        data: dataframe holding data to plot
        feature: column name of the chosen feature
        timestamp: column name of the timestamp associated with the feature
        tag_range: range of a feature to show on a plot
        target: column name of the target variable
        title: title of the chart
        labels_length_limit: limits feature name to `name[:feature_name_limit]...`
            in case it's too long
        fig_params: kwargs for plotly chart function
        layout_params: dictionary containing keys and values for updating
            the plotly fig layout

    Returns:
        plotly feature overview chart
    """

    check_data(data, timestamp, feature, target)

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{}, {}, {}],
            [
                {"colspan": 2, "secondary_y": True},
                None,
                {"colspan": 1, "type": "table"},
            ],
        ],
        subplot_titles=(
            "Boxplot",
            "Histogram",
            "Scatter vs target",
            "Timeseries",
            "Summary statistics",
        ),
        column_widths=[0.33, 0.33, 0.33],
    )

    feature_compact_name = _wrap_string(feature, labels_length_limit)
    legend_group = "group"

    _add_overview_first_row_plots(
        fig,
        feature=data[feature],
        feature_compact_name=feature_compact_name,
        legend_group=legend_group,
        target=(data[target] if target else None),
    )

    _add_overview_second_row_plot(
        fig,
        feature=data[feature],
        timestamp=data[timestamp],
        feature_compact_name=feature_compact_name,
        legend_group=legend_group,
    )

    # NOTE: The ranges _must_ be added
    # - after the plots have been generated
    # - before the table is added
    # This is due to a bug (feature?:D) in plotly, where vlines and hlines can be
    # added in a make_subplots if there are no go.Table present.
    # Otherwise it generates this cryptic and difficult to debug error.
    # ```
    # _plotly_utils.exceptions.PlotlyKeyError: Invalid property specified for object of
    #  type plotly.graph_objs.Table: 'xaxis'
    #
    # Did you mean "cells"?
    # ````
    # See here https://github.com/plotly/plotly.py/issues/3424
    if tag_range is not None:
        _add_ranges(fig, tag_range, target)

    stats = _generate_stats(data, feature)
    _add_overview_second_row_table(fig, stats)

    target_compact_name = _wrap_string(target, labels_length_limit)
    _update_overview_layout(
        fig,
        title,
        feature,
        feature_compact_name,
        target_compact_name,
    )
    apply_chart_style(
        fig=fig,
        title=title,
        fig_params=fig_params,
        layout_params=layout_params,
    )
    return fig


def _add_ranges(fig: go.Figure, tag_range: TRange, target: tp.Optional[str]) -> None:
    """Adds ranges to the feature overview plots"""
    _add_range(fig, row=1, col=1, tag_range=tag_range, direction="h")
    _add_range(fig, row=1, col=2, tag_range=tag_range, direction="v")
    _add_range(fig, row=2, col=1, tag_range=tag_range, direction="h")
    if target:
        _add_range(fig, row=1, col=3, tag_range=tag_range, direction="v")


def plot_feature_overviews(
    data: pd.DataFrame,
    features: tp.Union[str, tp.Iterable[str]],
    timestamp: str,
    tag_ranges: tp.Optional[TRangeOrDictWithRange] = None,
    target: tp.Optional[str] = None,
    title: tp.Optional[str] = None,
    labels_length_limit: int = 20,
    fig_params: tp.Optional[TDict] = None,
    layout_params: tp.Optional[TDict] = None,
) -> tp.Union[go.Figure, tp.Dict[str, go.Figure]]:
    """
    Create plots a collection of plots representing chosen feature "overview".
    This includes a boxplot and histogram to understand the distribution of values,
    a scatter-plot vs the `target` variable, and a time-series plot of the `feature` and
    `target`.

    Args:
        data: dataframe holding data to plot
        features: feature or features to show
        tag_ranges: tag's or tags' ranges (min, max) to show
        timestamp: column name of the timestamp associated with the feature
        target: column name of the target variable
        title: title of the chart
        labels_length_limit: limits feature name to `name[:feature_name_limit]...`
            in case it's too long
        fig_params: kwargs for plotly chart function
        layout_params: dictionary containing keys and values for updating
            the plotly fig layout

    Returns:
        dictionary containing plotly feature overview chart
    """
    if timestamp not in data.columns and timestamp in data.index.names:
        data = data.reset_index()
    check_data(data, timestamp, features, target)

    if isinstance(features, str):
        features = [features]
    elif not isinstance(features, list):
        features = list(features)

    tag_ranges = _validate_tag_ranges(features, tag_ranges)

    figs = {
        feature: _plot_feature_overview(
            data=data,
            feature=feature,
            timestamp=timestamp,
            target=target,
            tag_range=tag_ranges.get(feature),
            title=title,
            labels_length_limit=labels_length_limit,
            fig_params=fig_params,
            layout_params=layout_params,
        )
        for feature in features
    }
    return figs if len(features) > 1 else figs[features[0]]


def _validate_tag_ranges(
    features: tp.List[str],
    tag_ranges: tp.Optional[TRangeOrDictWithRange] = None,
) -> tp.Dict[str, TRange]:
    if tag_ranges is None:
        tag_ranges = {}
    elif isinstance(tag_ranges, tuple) and len(features) == 1:
        tag_ranges = {features[0]: tag_ranges}
    elif isinstance(tag_ranges, tuple):
        raise ValueError("`tag_ranges` can be `tuple` only when one feature is passed")
    return tag_ranges


def _make_limited_string(
    string: tp.Optional[str],
    max_length: int = 25,
    placeholder: str = "...",
) -> tp.Optional[str]:
    if string is None or len(string) <= max_length:
        return string
    return textwrap.shorten(string, max_length, placeholder=placeholder)


def _wrap_string(
    string: tp.Optional[str],
    max_length: int = 25,
    wrap_by: str = "<br>",
    replace_with_hyphens: tp.Iterable[str] = ("_",),
) -> tp.Optional[str]:
    if string is None or len(string) <= max_length:
        return string

    # we will use `textwrap.wrap` to perform wrapping on mutated string
    # and then use that to collect wrapped from initial string
    initial_string = string
    for replace in replace_with_hyphens:
        string = string.replace(replace, " ")
    current_index = 0
    wrapping = []
    for wrapped_seq in textwrap.wrap(string, max_length):
        wrapping.append(
            initial_string[current_index : current_index + len(wrapped_seq)],
        )
        current_index += len(wrapped_seq) + 1
    return wrap_by.join(wrapping)
