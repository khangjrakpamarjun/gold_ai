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
from plotly.subplots import make_subplots
from scipy import stats

from reporting.charts.utils import calculate_optimal_bin_width

_TRange = tp.Tuple[float, float]
_TPoint = tp.Tuple[float, float]
_TAlignmentLine = tp.Tuple[_TPoint, _TPoint]

PREDICTED = "predicted"
RESIDUAL = "residual"
ACTUAL = "actual"
PLOT_HEIGHT = 600
LEGEND_X_OFFSET, LEGEND_Y_OFFSET = 1, 1.1
SUBPLOTS_HORIZONTAL_SPACING = 0.03
SUBPLOTS_VERTICAL_SPACING = 0.09


def _get_color(name: tp.Optional[str] = None, opacity: float = 0.7) -> str:
    if name is None:
        return f"rgba(43, 75, 110, {opacity})"
    return {
        PREDICTED: f"rgba(184, 71, 93, {opacity})",
        RESIDUAL: f"rgba(148, 103, 189, {opacity})",
        ACTUAL: f"rgba(71, 184, 162, {opacity})",
    }[name]


def plot_actual_vs_residuals(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    timestamp_column: str,
    prediction_column: str,
    target_column: str,
) -> go.Figure:
    """
    Plots actual vs residuals dashboard with two rows (Train and Test)
    and three columns (Timeline, Distplot and Alignment)

    Args:
        train_data: train dataset with timestamp, target and prediction columns
        test_data: test dataset with timestamp, target and prediction columns
        timestamp_column: timestamp column name; can be of any dtype (datetime or just
            int)
        target_column: target column name
        prediction_column: prediction column name
    """

    residuals_column = "residuals"
    train_data[residuals_column] = (
        train_data[target_column] - train_data[prediction_column]
    )
    test_data[residuals_column] = (
        test_data[target_column] - test_data[prediction_column]
    )
    return _plot_actual_vs_feature(
        train_data,
        test_data,
        timestamp_column,
        residuals_column,
        RESIDUAL,
        target_column,
        is_residuals_mode=True,
        distplot_is_vertical=True,
        same_bin_size_for_all=False,
    )


def plot_actual_vs_predicted(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    timestamp_column: str,
    prediction_column: str,
    target_column: str,
) -> go.Figure:
    """
    Plots actual vs. predicted dashboard with two rows (Train and Test)
    and three columns (Timeline, Distplot and Alignment)

    Args:
        train_data: train dataset with timestamp, target and prediction columns
        test_data: test dataset with timestamp, target and prediction columns
        timestamp_column: timestamp column name; can be of any dtype (datetime or just
            int)
        target_column: target column name
        prediction_column: prediction column name
    """

    return _plot_actual_vs_feature(
        train_data,
        test_data,
        timestamp_column,
        prediction_column,
        PREDICTED,
        target_column,
        is_residuals_mode=False,
        distplot_is_vertical=False,
        same_bin_size_for_all=True,
    )


def _plot_one_row_of_actual_vs_feature_graphs(
    fig: go.Figure,
    data: pd.DataFrame,
    timestamp_column: str,
    feature_column: str,
    target_column: str,
    feature_name: str,
    is_residuals_mode: bool,
    show_legend: bool,
    row: int,
    alignment_plot_axis_id,
    distplot_is_vertical: bool,
    histogram_bin_size: tp.Optional[tp.Union[float, int]],
) -> None:
    """
    Plots one row of the actual vs feature graphs for a given data (train/test/other.)

    One row of the actual vs features graphs consists of
    - time series plot
    - distribution plot
    - actual and feature alignment
    """
    data = data.sort_values(timestamp_column)
    feature = data[feature_column]
    timestamp = data[timestamp_column]
    target = data[target_column]

    series_to_show = [(feature_name, feature)]
    # in residuals mode we show only residuals timeline and distribution
    if not is_residuals_mode:
        series_to_show.append((ACTUAL, target))

    for name, series in series_to_show:
        fig.add_trace(
            _get_timeline(timestamp, series, name, show_legend=show_legend),
            row=row,
            col=1,
        )
        fig.add_traces(
            data=_get_distplot_traces(
                series,
                name,
                is_vertical=distplot_is_vertical,
                histogram_bin_size=histogram_bin_size,
            ),
            rows=row,
            cols=2,
        )
    _plot_actual_and_feature_alignment(
        fig,
        target,
        feature,
        feature_name,
        row=row,
        col=3,
        axis_index=alignment_plot_axis_id,
        perfect_alignment_line=_define_perfect_alignment_line(
            target,
            is_residuals_mode,
        ),
    )


def _determine_bin_size_parameter(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_column: str,
) -> float:
    """
    Used to calculate a unified bin size to use for all the plots in
     ``_plot_actual_vs_feature``.

    Finds a unified ideal bin size for all the feature data, including both train and
     test data.
    """
    all_data = pd.concat([train_data, test_data])
    series_for_bin_size_calculation = all_data[feature_column]
    return calculate_optimal_bin_width(series_for_bin_size_calculation)


def _plot_actual_vs_feature(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    timestamp_column: str,
    feature_column: str,
    feature_name: str,
    target_column: str,
    is_residuals_mode: bool,
    distplot_is_vertical: bool,
    same_bin_size_for_all: bool,
) -> go.Figure:
    """
    Creates a figure with two rows and three columns.
    First row contain plots for train, second for test data.
    Columns of each row contain following comparisons:
        * timeline
        * distribution
        * alignment
    """
    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[[{}, {}, {}], [{}, {}, {}]],
        column_widths=[0.5, 0.25, 0.25],
        column_titles=[
            f"<b>{title}</b>" for title in ("Timeline", "Distplot", "Alignment")
        ],
        horizontal_spacing=SUBPLOTS_HORIZONTAL_SPACING,
        vertical_spacing=SUBPLOTS_VERTICAL_SPACING,
    )
    # One size to rule them all: better if same bin size is used for all plots
    histogram_bin_size = (
        _determine_bin_size_parameter(train_data, test_data, feature_column)
        if same_bin_size_for_all
        else None
    )
    # For row 1 and 2, info of alignment_plot_axis_id, data, and show_legend
    rows_info = zip((1, 2), (3, 6), (train_data, test_data), (True, False))
    for row, alignment_plot_axis_id, data, show_legend in rows_info:
        _plot_one_row_of_actual_vs_feature_graphs(
            fig,
            data,
            timestamp_column,
            feature_column,
            target_column,
            feature_name,
            is_residuals_mode,
            show_legend,
            row,
            alignment_plot_axis_id,
            distplot_is_vertical=distplot_is_vertical,
            histogram_bin_size=histogram_bin_size,
        )

    _update_layout(
        fig=fig,
        actuals_range=_get_ranges(train_data, test_data, target_column),
        feature_range=_get_ranges(train_data, test_data, feature_column),
        feature_name=feature_name,
        link_alignment_x_axis_to_y=not is_residuals_mode,
        distplot_is_vertical=distplot_is_vertical,
    )
    return fig


def _define_perfect_alignment_line(
    target: pd.Series,
    is_residuals_mode: bool,
) -> _TAlignmentLine:
    """Gives the coordinates of the two extreme points of the x=y line.

    This line is useful to help comparing x and y visually.
    """
    min_target = min(target)
    max_target = max(target)
    if is_residuals_mode:
        return (min_target, 0), (max_target, 0)
    return (min_target, min_target), (max_target, max_target)


def _update_axes_ranges(
    fig: go.Figure,
    rows: tp.Tuple[int, ...],
    feature_range: _TRange,
    actuals_range: _TRange,
    distplot_is_vertical: bool = False,
) -> None:
    """Updates the axes ranges

    Modifies ``fig`` in place.
    """
    for row in rows:
        fig.update_yaxes(range=feature_range, row=row, col=1)
        if distplot_is_vertical:
            fig.update_xaxes(range=feature_range, row=row, col=2)
            fig.update_yaxes(autorange=True, row=row, col=2)
        else:
            fig.update_yaxes(range=feature_range, row=row, col=2)
            fig.update_xaxes(autorange=True, row=row, col=2)
        fig.update_yaxes(range=feature_range, constrain="domain", row=row, col=3)
        fig.update_xaxes(range=actuals_range, constrain="domain", row=row, col=3)


def _remove_distplot_y_ticks(fig: go.Figure, rows) -> None:
    """Remove distplot y-ticks

    Modifies ``fig`` in place.
    """
    for row in rows:
        fig.update_yaxes(showticklabels=False, row=row, col=2)


# Naming makes meaning clearer, so ok to ignore WPS118 in the next line
def _linx_x_axes_between_train_test_alignment_plots(  # noqa: WPS118
    fig: go.Figure,
) -> None:
    """Links x-axes between train/test alignment plots

    Modifies ``fig`` in place.
    """
    fig.update_xaxes(row=2, col=3, matches="x3")


def _link_plots(
    fig: go.Figure,
    rows: tp.Tuple[int, ...],
    ref_axis: str,
    link_alignment_x_axis_to_y: bool,
    distplot_is_vertical: bool = False,
) -> None:
    """Links ref_axis across plots

    Modifies ``fig`` in place.
    """
    for row in rows:
        fig.update_yaxes(row=row, col=1, matches=ref_axis)
        if distplot_is_vertical:
            fig.update_xaxes(row=row, col=2, matches=ref_axis)
        else:
            fig.update_yaxes(row=row, col=2, matches=ref_axis)
        fig.update_yaxes(row=row, col=3, matches=ref_axis)
        if link_alignment_x_axis_to_y:
            fig.update_xaxes(row=row, col=3, matches=ref_axis)


def _add_train_test_labels_for_rows(fig: go.Figure, rows: tp.Tuple[int, ...]) -> None:
    """Add train & test labels for rows

    Modifies ``fig`` in place.
    """
    for row, title in zip(rows, ("Train", "Test")):
        fig.update_yaxes(
            title={"text": f"<b>{title}</b>", "standoff": 0, "font_size": 16},
            row=row,
            col=1,
        )


def _add_target_alignment_labels(
    fig: go.Figure,
    rows: tp.Tuple[int, ...],
    feature_name: str,
) -> None:
    """
    Add target alignment labels

    Modifies ``fig`` in place.
    """
    for row in rows:
        label_config = {
            "side": "right",
            "title": {"standoff": 3, "font_size": 12},
        }
        fig.update_xaxes(title_text=ACTUAL, **label_config, row=row, col=3)
        fig.update_yaxes(title_text=feature_name, **label_config, row=row, col=3)


def _update_target_vs_feature_plot_layout(
    fig: go.Figure,
    feature_name: str,
) -> None:
    """Updates the layout of the Target vs. Actual plot

    Modifies ``fig`` in place.
    """
    fig.update_layout(
        height=PLOT_HEIGHT,
        title_text=f"Target {feature_name.capitalize()} vs. Actual",
        legend=dict(  # noqa: C408
            orientation="h",
            x=LEGEND_X_OFFSET,
            y=LEGEND_Y_OFFSET,
            xanchor="right",
            yanchor="bottom",
        ),
        barmode="overlay",
        hoverlabel_align="right",
    )
    fig.update_annotations(yshift=10)
    fig.update_xaxes(zeroline=False, tickfont_size=9)
    fig.update_yaxes(zeroline=False, tickfont_size=9)


def _update_layout(
    fig: go.Figure,
    actuals_range: _TRange,
    feature_range: _TRange,
    feature_name: str,
    link_alignment_x_axis_to_y: bool,
    distplot_is_vertical: bool = False,
) -> None:
    """Modifies ``fig`` in place."""
    _update_target_vs_feature_plot_layout(fig=fig, feature_name=feature_name)

    rows = (1, 2)
    _update_axes_ranges(
        fig=fig,
        rows=rows,
        feature_range=feature_range,
        actuals_range=actuals_range,
        distplot_is_vertical=distplot_is_vertical,
    )
    _remove_distplot_y_ticks(fig=fig, rows=rows)
    _linx_x_axes_between_train_test_alignment_plots(fig=fig)
    _link_plots(
        fig=fig,
        rows=rows,
        ref_axis="y1",
        link_alignment_x_axis_to_y=link_alignment_x_axis_to_y,
        distplot_is_vertical=distplot_is_vertical,
    )
    _add_train_test_labels_for_rows(fig=fig, rows=rows)
    _add_target_alignment_labels(fig=fig, rows=rows, feature_name=feature_name)


def _get_ranges(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    column: str,
    relative_offset_from_range: float = 0.04,
) -> _TRange:
    """
    returns min and max values that define a nice plotting range for train and test
     values of the feature named ``column``.

    Notes:
        For given feature (``column`` from both ``train_data`` and ``test_data``)
         determines a "good plotting range" that works when showing both train and test
         data.
        It does this in the following way:
        - add offset ouside the data points
        - estimate such offset as a percentage of the range
        - (and estimate such range as the greatest between the train and test data
         range)
    """
    train_series = train_data[column]
    test_series = test_data[column]
    q_min, q_max = 0.05, 0.95
    range_ = max(
        train_series.quantile(q_max) - train_series.quantile(q_min),
        test_series.quantile(q_max) - test_series.quantile(q_min),
    )
    offset = relative_offset_from_range * range_
    return (
        min(train_series.min(), test_series.min()) - offset,
        max(train_series.max(), test_series.max()) + offset,
    )


def _plot_actual_and_feature_alignment(
    fig: go.Figure,
    target: pd.Series,
    feature: pd.Series,
    feature_name: str,
    perfect_alignment_line: _TAlignmentLine,
    row: int,
    col: int,
    axis_index: int,
    hover_precision: int = 2,
) -> None:
    formatting = _get_float_precision_formatting(hover_precision)
    fig.add_trace(
        go.Scatter(
            x=target,
            y=feature,
            mode="markers",
            name="<br>".join([ACTUAL, feature_name]),
            showlegend=False,
            marker={"color": _get_color(opacity=0.5)},
            hovertemplate=(
                # String concatenation instead of f-string used to avoid issues
                # when the string has curly braces inside
                "%{x"
                + formatting
                + "}<br>"
                + "%{y"
                + formatting
                + "}"  # noqa: WPS336
            ),
        ),
        row=row,
        col=col,
    )

    # Add diagonal line
    (from_x, from_y), (to_x, to_y) = perfect_alignment_line
    fig.add_shape(
        type="line",
        xref=f"x{axis_index}",
        yref=f"y{axis_index}",
        x0=from_x,
        y0=from_y,
        x1=to_x,
        y1=to_y,
        line=dict(dash="dot", color="white"),  # noqa: C408
    )


def _get_timeline(
    timestamps: pd.Series,
    series: pd.Series,
    name: str,
    show_legend: bool = True,
    hover_precision: int = 2,
) -> go.Scatter:
    # todo: Tickformatstops to customize for different zoom levels
    formatting = _get_float_precision_formatting(hover_precision)
    return go.Scatter(
        x=timestamps,
        y=series,
        name=name,
        marker={"color": _get_color(name)},
        showlegend=show_legend,
        legendgroup=name,
        line={"width": 1.5},
        # String concatenation instead of f-string used to avoid issues
        # when the string has curly braces inside
        hovertemplate="%{y" + formatting + "}<br>%{x}",  # noqa: WPS336
    )


def _get_distplot_traces(
    series: pd.Series,
    name: str,
    is_vertical: bool = True,
    relative_offset: float = 0.15,
    is_categorical: bool = False,
    histogram_bin_size: tp.Optional[tp.Union[float, int]] = None,
) -> tp.List[tp.Union[go.Histogram, go.Scatter]]:
    traces = []
    hist_range = (series.min(), series.max())
    if not is_categorical:
        hist_range = (
            (1 - relative_offset) * hist_range[0],
            (1 + relative_offset) * hist_range[1],
        )
    traces.append(
        _get_hist(
            series,
            name,
            hist_range,
            is_vertical,
            is_categorical,
            bin_size=histogram_bin_size,
        )
    )
    if not is_categorical:
        traces.append(_get_density_line(series, name, hist_range, is_vertical))
    return traces


def _get_hist(
    series: pd.Series,
    name: str,
    hist_range: _TRange,
    is_vertical: bool,
    is_categorical: bool,
    bin_size: tp.Optional[tp.Union[float, int]] = None,
) -> go.Histogram:
    if is_categorical:
        data_kwargs = (
            {"x": series, "hovertemplate": "%{x}"}
            if is_vertical
            else {"y": series, "hovertemplate": "%{y}"}
        )
    else:
        bin_size = calculate_optimal_bin_width(series) if bin_size is None else bin_size
        range_start, range_end = hist_range
        bins_config = {
            "start": range_start,
            "end": range_end,
            "size": bin_size,
        }
        data_kwargs = (
            {"x": series, "xbins": bins_config, "hovertemplate": "%{x}"}
            if is_vertical
            else {"y": series, "ybins": bins_config, "hovertemplate": "%{y}"}
        )
    return go.Histogram(
        histnorm="probability density",
        legendgroup=name,
        marker={"color": _get_color(name, opacity=0.5)},
        name=name,
        showlegend=False,
        **data_kwargs,
    )


def _get_density_line(
    series: pd.Series,
    name: str,
    hist_range: _TRange,
    is_vertical: bool,
    n_points_per_kde: int = 500,
    hover_precision: int = 2,
) -> go.Scatter:
    hist_domain = np.linspace(*hist_range, n_points_per_kde)
    kde = stats.gaussian_kde(series)(hist_domain)
    formatting = _get_float_precision_formatting(hover_precision)
    data_kwargs = (
        {
            "x": hist_domain,
            "y": kde,
            # String concatenation instead of f-string used to avoid issues
            # when the string has curly braces inside
            "hovertemplate": "%{x" + formatting + "}",  # noqa: WPS336
        }
        if is_vertical
        else {
            "x": kde,
            "y": hist_domain,
            # String concatenation instead of f-string used to avoid issues
            # when the string has curly braces inside
            "hovertemplate": "%{y" + formatting + "}",  # noqa: WPS336
        }
    )
    return go.Scatter(
        name=name,
        legendgroup=name,
        showlegend=False,
        marker={"color": _get_color(name, opacity=1.0)},
        **data_kwargs,
    )


def _get_float_precision_formatting(precision: int) -> str:
    return f":.{precision}f"
