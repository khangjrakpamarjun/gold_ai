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
import math
import typing as tp

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from reporting.api.types import ShapExplanation
from reporting.charts.utils import calculate_optimal_bin_width

from ._shap_lib_utils import potential_interactions  # noqa: WPS436
from ._utils import (  # noqa: WPS436
    FeatureShapExplanation,
    extract_explanations_for_given_features,
    sort_features,
)

_TRange = tp.Tuple[float, float]
_TITLE_PREFIX = "%TITLE%"
_AUTO_COLOR_BY_ARG = "_auto"
_LINE_COLOR_NON_COLORED = "rgb(43, 75, 110)"
_COLOR_SCALE = ("rgb(56,138,243)", "rgb(234,51,86)")

LAYOUT_MARGIN_L = 130
LAYOUT_MARGIN_R = 130
LAYOUT_MARGIN_B = 80
LAYOUT_LEGEND_OFFSET_X = 1
LAYOUT_LEGEND_OFFSET_Y = 1.02

logger = logging.getLogger(__name__)


def plot_shap_dependency(  # noqa: WPS210
    features: tp.List[str],
    shap_explanation: ShapExplanation,
    order_by: tp.Optional[np.ndarray] = None,
    color_by: tp.Optional[str] = _AUTO_COLOR_BY_ARG,
    max_features_to_display: tp.Optional[int] = 20,
    n_columns: int = 2,
    subplot_height: tp.Optional[int] = 320,
    subplot_width: tp.Optional[int] = 370,
    horizontal_spacing_per_row: float = 0.8,
    vertical_spacing_per_column: float = 0.45,
) -> go.Figure:
    """
    Plots shap values of a feature against its values

    Args:
        features: feature names to draw dependency for
        shap_explanation: shap explanation
        order_by: array to DESCENDING sort features by;
            sorts by mean abs shap value by default
        n_columns: number of columns to show
        color_by: feature name of shap values to use for coloring scatter points by;
            `color_by` has to be present in `shap_explanation`;
            if `None` is provided, no coloring applied;
            if `"_auto"` is passed, we'll color each scatter by the feature that
            it has the most interaction with
            (note that we use shap lib approximation for those interactions);
        max_features_to_display: max features to show in explanation; show all if None
        subplot_height: each figure's subplot height
        subplot_width: each figure's subplot width
        horizontal_spacing_per_row: horizontal spacing between subplots
            in normalized plot coordinates
            (the value is divided by number of plots since the total plot size grows)
        vertical_spacing_per_column: vertical spacing between subplots
            in normalized plot coordinates
            (the value is divided by number of plots since the total plot size grows)
    """
    features = sort_features(features, order_by, shap_explanation, descending=True)
    if max_features_to_display is not None and len(features) > max_features_to_display:
        logger.info(
            f"Too many features provided. "
            f"Only first {max_features_to_display = } will be plot.",
        )
        features = features[:max_features_to_display]

    by_feature_explanation = extract_explanations_for_given_features(
        features,
        shap_explanation,
    )
    n_features = len(features)
    n_rows = math.ceil(n_features / n_columns)
    fig = make_subplots(
        cols=n_columns,
        rows=n_rows,
        figure=go.Figure(
            layout=_get_layout(n_rows * subplot_height, n_columns * subplot_width),
        ),
        horizontal_spacing=horizontal_spacing_per_row / n_columns,
        vertical_spacing=vertical_spacing_per_column / n_rows,
        subplot_titles=[f"{_TITLE_PREFIX}{feat}" for feat in features],
    )

    explanation: FeatureShapExplanation
    for plot_index, (feature, explanation) in enumerate(by_feature_explanation.items()):
        row = plot_index // n_columns + 1
        column = plot_index % n_columns + 1
        coloring = _get_coloring(
            fig,
            color_by,
            explanation,
            shap_explanation,
            row,
            column,
        )
        y_axis_range = _get_y_axis_range(explanation.values)
        _add_scatter(
            fig=fig,
            feature_values=explanation.data,
            feature_shaps=explanation.values,
            coloring=coloring,
            row=row,
            column=column,
        )
        _add_histogram(fig, explanation.data, y_axis_range, row, column)
        _update_subplot_layout(fig, feature, y_axis_range, row, column)
    return fig


def _update_subplot_layout(
    fig: go.Figure,
    feature: str,
    y_axis_range: _TRange,
    row: int,
    column: int,
    title_offset: float = 0.005,
) -> None:
    fig.update_xaxes(
        title="feature values",
        showline=True,
        linecolor="black",
        ticks="outside",
        hoverformat=".2f",
        row=row,
        col=column,
    )
    fig.update_yaxes(
        title="SHAP values",
        showline=True,
        linecolor="black",
        ticks="outside",
        range=y_axis_range,
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="lightgrey",
        row=row,
        col=column,
    )
    subplot_title = next(
        fig.select_annotations(selector={"text": f"{_TITLE_PREFIX}{feature}"}),
    )
    subplot_text_without_prefix = subplot_title.text.lstrip(_TITLE_PREFIX)
    subplot_title.text = f"<b>{subplot_text_without_prefix}</b>"
    subplot_title.y += title_offset


def _add_histogram(
    fig: go.Figure,
    feature_values: np.ndarray,
    y_axis_range: _TRange,
    row: int,
    colum: int,
) -> None:
    bar_centers, bar_counts, bar_heights = _get_histogram_data(
        feature_values,
        y_axis_range,
    )
    is_first_plot = row == 1 and colum == 1
    fig.add_trace(
        go.Bar(
            x=bar_centers,
            y=bar_heights,
            customdata=bar_counts,
            name="feature hist",
            base=min(y_axis_range),
            marker=dict(  # noqa: C408
                color="lightgrey",
                line_width=0,
            ),
            opacity=1,
            hovertemplate="%{customdata}",
            legendgroup="histogram",
            showlegend=is_first_plot,
        ),
        row=row,
        col=colum,
    )


def _add_scatter(
    fig: go.Figure,
    feature_values: np.ndarray,
    feature_shaps: np.ndarray,
    coloring: tp.Dict[str, tp.Any],
    row: int,
    column: int,
) -> None:
    fig.add_trace(
        # todo: there's a bug with scattergl
        #  when number of plots is ~20 scatters are shifted up, away from axes.
        #  we can mitigate this by using scatter in case of n_features > XX.
        #  this will also require removal of color bar border,
        #  because that exists by default for go.Scatter
        go.Scattergl(
            x=feature_values,
            y=feature_shaps,
            mode="markers",
            name="feature value<br>shap value",
            hovertemplate="%{x:.2f}<br>%{y:.2f}",
            marker=coloring,
            showlegend=False,
        ),
        row=row,
        col=column,
    )


def _get_histogram_data(
    # `values`` seems an appropriate name in this case
    values: np.ndarray,
    y_axis_range: _TRange,  # noqa: WPS110
) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gets positions, counts and height of the histogram bars for the ``values``.

    When generates the bars, only includes these with strictly more than zero points
     (if the bar would have zero height, then there is no bar).
    It returns
    - bar "centers" (called like this, but are the left side at the moment, see TODO
     below)
    - bar counts: the number of values in each bin
    - bar heights: a modified version of the bar coints, that accounts for the fact that
     the bars are only supposed to fill part of the graph
    """
    values_dropna = values[~np.isnan(values)]
    bar_counts, bin_edges = np.histogram(
        values_dropna,
        bins=_get_optimal_bins_number(values_dropna),
    )
    # TODO: Check if a /2 is missing after the parenthesis in the next line
    #  I think this is part of two bugs that compensate
    #  - this is the left side of the bar, not the center
    #  - by default plotly aligns vertical bars on the left, horizontal bar at center
    #  The solution is to
    #  - make this function return the actual centers
    #  - use the x-anchor property set to center when drawing the go.Bar
    #  N.B.: Need to veryfiy if/that this works first
    bar_centers = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1])
    bar_centers = bar_centers[bar_counts > 0]
    bar_counts = bar_counts[bar_counts > 0]
    portion_of_axis_to_occupy = 0.3
    bar_heights = _scale_plot_data_to_fit_part_of_chart(
        hist_counts=bar_counts,
        axis_range=y_axis_range,
        portion_of_axis_to_occupy=portion_of_axis_to_occupy,
        shift_based_on_axis_range=False,
    )
    return bar_centers, bar_counts, bar_heights


def _scale_plot_data_to_fit_part_of_chart(
    hist_counts: np.ndarray,
    axis_range: _TRange,
    portion_of_axis_to_occupy: float = 0.2,
    shift_based_on_axis_range: bool = True,
) -> np.ndarray:
    scale_up_to_max_values = max(axis_range) / hist_counts.max()
    scaled_values = portion_of_axis_to_occupy * scale_up_to_max_values * hist_counts
    if shift_based_on_axis_range:
        return scaled_values + min(axis_range)
    return scaled_values


def _get_y_axis_range(shap_values: np.ndarray) -> _TRange:
    shap_values_range = shap_values.max() - shap_values.min()
    rel_of_range_offset = 0.05
    offset = rel_of_range_offset * shap_values_range
    y_axis_range = (shap_values.min() - offset, shap_values.max() + offset)
    return y_axis_range  # noqa: WPS331  # Naming makes meaning clearer


def _get_layout(
    height: int,
    width: int,
) -> go.Layout:
    return go.Layout(
        height=height,
        width=width,
        bargap=0,
        plot_bgcolor="rgba(0,0,0,0)",
        title="SHAP Values Dependence Plot",
        legend=dict(  # noqa: C408
            orientation="h",
            x=LAYOUT_LEGEND_OFFSET_X,
            y=LAYOUT_LEGEND_OFFSET_Y,
            xanchor="left",
            yanchor="bottom",
        ),
        margin=dict(  # noqa: C408
            l=LAYOUT_MARGIN_L,
            r=LAYOUT_MARGIN_R,
            b=LAYOUT_MARGIN_B,
        ),
        hovermode="x",
    )


def _get_color_bar(fig: go.Figure, row: int, column: int, color_by: tp.Optional[str]):
    _, xaxis_domain_end = next(fig.select_xaxes(row=row, col=column)).domain
    yaxis_domain = next(fig.select_yaxes(row=row, col=column)).domain
    yaxis_domain_center = sum(yaxis_domain) / 2
    yaxis_domain_length = yaxis_domain[1] - yaxis_domain[0]
    x_offset = 0.01

    color_bar = dict(  # noqa: C408
        title=color_by,
        titleside="right",
        tickmode="array",
        ticks="outside",
        thickness=5,
        len=yaxis_domain_length,
        x=xaxis_domain_end + x_offset,
        y=yaxis_domain_center,
    )
    return color_bar  # noqa: WPS331  # Naming makes meaning clearer


def _get_coloring(
    fig: go.Figure,
    color_by: tp.Optional[str],
    feature_shap_explanation: FeatureShapExplanation,
    shap_explanation: ShapExplanation,
    row: int,
    column: int,
) -> tp.Dict[str, tp.Any]:
    if color_by is None:
        return dict(color=_LINE_COLOR_NON_COLORED)  # noqa: C408
    elif color_by == _AUTO_COLOR_BY_ARG:
        most_interacting_feature_index = potential_interactions(
            feature_shap_explanation,
            shap_explanation,
        )[0]
        color_by = shap_explanation.feature_names[most_interacting_feature_index]
    else:
        # validate requested `color_by` is present
        if color_by not in shap_explanation.feature_names:
            raise ValueError(
                f"Couldn't find requested shap values to color_by ({color_by})",
            )

    explanation_color_by = extract_explanations_for_given_features(
        [color_by],
        shap_explanation,
    )[color_by]
    color_bar = _get_color_bar(fig, row, column, color_by)
    marker = dict(  # noqa: C408
        color=explanation_color_by.data,
        colorscale=_COLOR_SCALE,
        opacity=1,
        colorbar=color_bar,
    )
    return marker  # noqa: WPS331  # Naming makes meaning clearer


def _get_optimal_bins_number(data: np.ndarray) -> int:
    """Returns the number of bins needed in order to have bins of optimal width.

    It calculates the optimal bin width with ``calculate_optimal_bin_width``, then just
     divides the data range by that to get the ideal number of bins (then rounds it up).
    """
    optimal_bin_width = calculate_optimal_bin_width(data)
    values_range = data.max() - data.min()
    return math.ceil(values_range / optimal_bin_width)
