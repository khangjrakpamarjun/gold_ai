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

import logging
import types
import typing as tp
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from reporting.charts.primitive import CodePlot, plot_code, plot_string

logger = logging.getLogger(__name__)

MAX_PERIODS_THRESHOLD = 250

HOVER_TEMPLATE_SPLIT_TIMELINE = "%{x}<br>%{text}<extra></extra>"
HOVER_TEMPLATE_REFERENCE = "split set: %{customdata}<extra></extra>"

HEIGHT = 400

TEST_LABEL = "Test"
TRAIN_LABEL = "Train"

TITLE = "Train/Test Split Representation"
SPLIT_TIMELINE_LABEL = "split type<br>timeline"
COLOR_LINE = "grey"  # "#2B3467"
SPLIT_COLOR_MAP = types.MappingProxyType(
    {
        TRAIN_LABEL: "rgb(31, 119, 180)",  # "blue",  # "#EB455F"
        TEST_LABEL: "rgb(255, 127, 14)",  # "red",  # "#0081B4"
    }
)
SPLIT_COLUMN = "split"
COLOR_GRID = "black"
COLOR_BACKGROUND = "rgba(0,0,0,0)"


_DEFAULT_PLOT_TITLE_SIZE = 20
_DEFAULT_OPACITY = 0.2
_DEFAULT_SUBPLOT_NORMALIZED_VERTICAL_SPACING = 0.2

_LabelType = tp.TypeVar("_LabelType", int, str)
ListOfPeriodsStartEnd = tp.List[tp.Tuple[int, int]]


def _get_consecutive_periods(
    split_labels: tp.Iterable[_LabelType],
) -> tp.Dict[_LabelType, ListOfPeriodsStartEnd]:
    """
    Evaluates periods of consecutive labels.
    We do forward and then backward propagation of known values to fill nan.
    """

    split_labels = pd.Series(split_labels).ffill().bfill().values
    if not len(split_labels):
        return {}

    # last point of period included
    change_points = np.where(split_labels[:-1] != split_labels[1:])[0]

    consecutive_periods = defaultdict(list)
    period_start = 0
    for point in change_points:
        current_period = split_labels[period_start]
        period_end = point + 1
        consecutive_periods[current_period].append((period_start, period_end))
        period_start = period_end
    current_period = split_labels[period_start]
    consecutive_periods[current_period].append((period_start, len(split_labels)))
    return dict(consecutive_periods)


def plot_validation_representation(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    reference_column: str,
    timestamp_column: str,
) -> go.Figure:
    data = pd.concat([train_data, test_data], ignore_index=True).sort_values(
        timestamp_column
    )
    split_labels = (
        data[timestamp_column]
        .isin(train_data[timestamp_column])
        .map({True: TRAIN_LABEL, False: TEST_LABEL})
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.9, 0.1],
        vertical_spacing=_DEFAULT_SUBPLOT_NORMALIZED_VERTICAL_SPACING,
    )
    _add_reference_line(fig, data, split_labels, reference_column, timestamp_column)
    _add_lower_navigation_with_colored_instances(
        fig,
        train_data,
        test_data,
        timestamp_column,
    )
    _add_split_periods_highlights_to_reference_line(
        fig,
        data,
        split_labels,
        timestamp_column,
    )
    _update_layout(fig)
    return fig


def plot_split_details(train_data: pd.DataFrame, test_data: pd.DataFrame) -> go.Figure:
    train_size = train_data.shape[0]
    test_size = test_data.shape[0]
    return plot_string(
        title="Split Details",
        text=f"train size: {train_size}\ntest size: {test_size}",
        title_size=_DEFAULT_PLOT_TITLE_SIZE,
    )


def _add_reference_line(
    fig: go.Figure,
    data: pd.DataFrame,
    split_labels: pd.Series,
    reference_column: str,
    timestamp_column: str,
) -> None:
    fig.add_traces(
        [
            go.Scattergl(
                x=data[timestamp_column],
                y=data[reference_column],
                customdata=split_labels,
                mode="lines",
                line=dict(width=1, color=COLOR_LINE),  # noqa: C408
                hovertemplate=HOVER_TEMPLATE_REFERENCE,
                name=reference_column,
                showlegend=False,
            ),
        ],
        rows=1,
        cols=1,
    )


def _plot_split_dataset(dataset: pd.DataFrame, timestamp_column: str, label: str):
    """
    Makes a scatter plot compatible with the usage for either train or test set
    in lower navigation with colored instances.

    Created as a helper function for
    ``_add_add_lower_navigation_with_colored_instances``.
    """
    timestamp = dataset[timestamp_column]
    marker_style = dict(  # noqa: C408
        symbol="line-ns",
        line_width=2,
        line_color=SPLIT_COLOR_MAP[label],
        color=SPLIT_COLOR_MAP[label],
    )
    return go.Scattergl(
        x=timestamp,
        y=[SPLIT_TIMELINE_LABEL for _ in timestamp],
        text=[label for _ in timestamp],
        hovertemplate=HOVER_TEMPLATE_SPLIT_TIMELINE,
        mode="markers",
        marker=marker_style,
        name=label,
        opacity=1,
    )


def _add_lower_navigation_with_colored_instances(
    fig: go.Figure,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    timestamp_column: str,
) -> None:
    """Adds a lower navigation with colored instances to ``fig``."""
    fig.add_traces(
        [
            _plot_split_dataset(
                dataset=dataset,
                timestamp_column=timestamp_column,
                label=split_label,
            )
            for split_label, dataset in {
                TRAIN_LABEL: train_data,
                TEST_LABEL: test_data,
            }.items()
        ],
        cols=1,
        rows=2,
    )


def _add_split_periods_highlights_to_reference_line(  # noqa: WPS118
    fig: go.Figure,
    data: pd.DataFrame,
    split_labels: pd.Series,
    timestamp_column: str,
) -> None:
    """Add highlights to the consecutive periods.

    Also, avoids to clutter the figure with highlights in the following way:
    - Periods are identified as eithertrain or test, based on the ``split_labels``
    - If the test periods are more than ``MAX_PERIODS_THRESHOLD``, then no highlights
       are added
    - If the test periods are not more that ``MAX_PERIODS_THRESHOLD``...:
        - ...but the total number of periods is not, then only the test periods are
           highlighted
        - ...and the total number of periods too, then all periods are highlighted
    """
    consecutive_periods = _get_consecutive_periods(split_labels)
    n_test_periods = len(consecutive_periods[TRAIN_LABEL])
    n_total_periods = len(consecutive_periods[TEST_LABEL]) + n_test_periods
    if n_test_periods > MAX_PERIODS_THRESHOLD:
        logger.warning(
            f"Too many consecutive splits found: {n_total_periods}. "
            f"Impossible to highlight any of those. "
            f"Skipping reference_column highlighting.",
        )
        consecutive_periods = {}
    elif n_test_periods <= MAX_PERIODS_THRESHOLD < n_total_periods:
        # try to reduce all rectangles to test periods only
        logger.warning(
            f"Too many consecutive splits found: {n_total_periods}. "
            f"Impossible to highlight all of those. "
            f"Will simplify highlighting to show only test periods.",
        )
        consecutive_periods.pop(TRAIN_LABEL)
    for split_label, periods in consecutive_periods.items():
        for period_start, period_end in periods:
            # reduce whitespaces between rectangles
            if period_end != data[timestamp_column].size:
                period_end += 1
            total_size_of_reference_plot = 0.72
            fig.add_shape(
                type="rect",
                x0=data[timestamp_column].iloc[period_start],
                x1=data[timestamp_column].iloc[period_end - 1],
                y0=1 - total_size_of_reference_plot,
                y1=1,
                layer="below",
                yref="paper",
                line=dict(width=0),  # noqa: C408
                fillcolor=SPLIT_COLOR_MAP[split_label],
                opacity=_DEFAULT_OPACITY,
            )


def _update_layout(fig: go.Figure) -> None:
    fig.update_layout(
        title=TITLE,
        plot_bgcolor=COLOR_BACKGROUND,
        height=HEIGHT,
        legend=dict(  # noqa: C408
            orientation="h",
            x=1,
            y=-0.1,
            xanchor="right",
            yanchor="top",
        ),
        hovermode="x unified",
        margin_pad=5,
        yaxis=dict(showgrid=False, zeroline=False),  # noqa: C408
        xaxis=dict(  # noqa: C408
            showgrid=True,
            rangeslider=dict(visible=True, thickness=0.1),  # noqa: C408
            gridcolor=COLOR_GRID,
        ),
        yaxis2=dict(  # noqa: C408
            tickvals=[SPLIT_TIMELINE_LABEL],
            showgrid=False,
            zeroline=False,
            fixedrange=True,
        ),
        xaxis2=dict(gridcolor=COLOR_GRID),  # noqa: C408
    )


def plot_consecutive_validation_periods(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    timestamp_column: str,
    timestamp_format: str = "%Y-%m-%d %H:%M:%S",  # noqa: WPS323
) -> tp.Dict[str, CodePlot]:
    data = pd.concat(
        [train_data[[timestamp_column]], test_data[[timestamp_column]]],
        ignore_index=True,
    )
    split_column = "split_labels"
    data[split_column] = np.concatenate(
        [
            [TRAIN_LABEL for _ in range(train_data.shape[0])],
            [TEST_LABEL for _ in range(test_data.shape[0])],
        ]
    )
    data = data.sort_values(timestamp_column)
    consecutive_period_timestamps = _get_consecutive_periods_timestamps(
        data,
        split_column,
        timestamp_column,
        timestamp_format,
    )
    return {
        name: plot_code("\n".join(periods), code_formatter=None, language="js")
        for name, periods in consecutive_period_timestamps.items()
    }


def _get_consecutive_periods_timestamps(
    data,
    split_column,
    timestamp_column,
    timestamp_format,
):
    """Gets the timestamps of the consecutive periods.

    Returns a dictionary where the key is the name of the consecutive period,
    and the value is a list with the formatted timestamps of the start and end of
    each of the periods in the consecutive periods.
    """
    consecutive_periods = _get_consecutive_periods(data[split_column])
    timestamps = pd.to_datetime(data[timestamp_column])
    consecutive_period_timestamps = {
        name: [
            '"{from_}", "{to}"'.format(
                from_=timestamps.iloc[period_start].strftime(timestamp_format),
                to=timestamps.iloc[period_end - 1].strftime(timestamp_format),
            )
            for period_start, period_end in periods
        ]
        for name, periods in consecutive_periods.items()
    }
    return consecutive_period_timestamps  # noqa: WPS331  # Naming makes meaning clearer
