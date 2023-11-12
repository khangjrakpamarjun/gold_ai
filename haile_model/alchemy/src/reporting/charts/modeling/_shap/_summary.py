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
import typing as tp

import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from reporting.api.types import ShapExplanation

from ._utils import (  # noqa: WPS436
    FeatureShapExplanation,
    extract_explanations_for_given_features,
    sort_features,
)

Y_AXIS_TITLE = "features"
_SHAP_VALUES_SUMMARY_ANNOTATIONS_Y = 0.55
_SHAP_VALUES_SUMMARY_ANNOTATIONS_X_SHIFT = -150
_SHAP_VALUES_SUMMARY_MARGIN_L = 210
# Adjusting the height of the summary plot, accounting for the fact that
# the height of the figure includes
# - the top margin (default: 100px)
# - the bottom margin (default: 80px)
# - the size of the title (default: ? might depend on the font size?)
#    Using a buffer of 20px, because anything less breaks the plot (with 2 features)
# - the size of the plot
# This prevents having too little vertical space left to draw the graph when there is a
#  small number of features
_SHAP_SUMMARY_HEIGHT_BUFFER = 200  # 100 + 80 + 20

logger = logging.getLogger(__name__)


def plot_shap_summary(
    features: tp.List[str],
    shap_explanation: ShapExplanation,
    order_by: tp.Optional[np.ndarray] = None,
    max_features_to_display: tp.Optional[int] = 15,
    opacity: float = 1.0,
    height_per_feature: int = 60,
    width: tp.Optional[int] = None,
) -> go.Figure:
    """
    Plots SHAP summary.
    Features' shap values are scatter plotted line by line.
    Points are jitter according to shap-values density.
    Coloring is done by feature's values.

    Args:
        features: features to plot summary for
        shap_explanation: shap explanation
        order_by: array to DESCENDING sort features by;
            sorts by mean abs shap value by default
        max_features_to_display: max features to show in explanation; show all if None
        opacity: opacity of scatter points
        height_per_feature: vertical size of each feature subplot
        width: figure's width
    """
    features = sort_features(features, order_by, shap_explanation, descending=True)

    if max_features_to_display is not None and len(features) > max_features_to_display:
        logger.info(
            f"Too many features provided. "
            f"Only first {max_features_to_display = } will be plot.",
        )
        features = features[:max_features_to_display]
    n_features = len(features)

    fig = make_subplots(
        rows=n_features,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0,
        y_title=Y_AXIS_TITLE,
    )

    by_feature_explanation = extract_explanations_for_given_features(
        features,
        shap_explanation,
    )

    explanation: FeatureShapExplanation
    for plot_index, (feature, explanation) in enumerate(by_feature_explanation.items()):
        _add_beeswarm_plot(
            fig=fig,
            shap_values=explanation.values,
            color_by=explanation.data,
            feature=feature,
            row_index=plot_index + 1,
            opacity=opacity,
        )

    _update_layout(fig, n_features, height_per_feature, width)
    _add_missing_data_legend(fig, by_feature_explanation)
    return fig


def _add_beeswarm_plot(
    fig: go.Figure,
    shap_values: np.ndarray,
    color_by: np.ndarray,
    feature: str,
    row_index: int,
    opacity: float,
) -> None:
    is_first_plot = row_index == 1
    feature_normalized = _get_minmax_normalized_feature(color_by)
    color_bar = _get_color_bar_argument(
        bar_range=(feature_normalized.min(), feature_normalized.max()),
        is_first_plot=is_first_plot,
    )
    fig.add_trace(
        go.Scattergl(
            x=shap_values,
            y=_jitter_points_based_on_distribution(shap_values),
            name=feature,
            showlegend=False,
            mode="markers",
            hovertemplate="%{x}",
            marker=dict(  # noqa: C408
                color=feature_normalized,
                showscale=is_first_plot,
                colorscale=["rgb(56,138,243)", "rgb(234,51,86)"],
                opacity=opacity,
                colorbar=color_bar,
            ),
        ),
        row=row_index,
        col=1,
    )
    # add feature name to the y-axis
    fig.update_yaxes(
        tickmode="array",
        tickvals=[0],
        ticktext=[feature],
        fixedrange=True,
        row=row_index,
        col=1,
    )


def _update_layout(
    fig: go.Figure,
    n_features: int,
    size_of_each_feature_plot: int,
    width: int,
) -> None:
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        title="SHAP Values Summary",
        autosize=False,
        height=size_of_each_feature_plot * n_features + _SHAP_SUMMARY_HEIGHT_BUFFER,
        width=width,
        margin_l=_SHAP_VALUES_SUMMARY_MARGIN_L,
    )
    fig.update_xaxes(zerolinecolor="black", zerolinewidth=1)
    fig.update_xaxes(
        title="SHAP value (impact on the model output)",
        ticks="inside",
        showline=True,
        linecolor="black",
        row=n_features,
    )
    fig.update_annotations(
        xshift=_SHAP_VALUES_SUMMARY_ANNOTATIONS_X_SHIFT,
        y=_SHAP_VALUES_SUMMARY_ANNOTATIONS_Y,
        selector={"text": Y_AXIS_TITLE},
    )


def _get_color_bar_argument(
    bar_range: tp.Tuple[float, float],
    is_first_plot: bool,
) -> tp.Optional[tp.Dict[str, tp.Any]]:
    if not is_first_plot:
        return None
    return dict(  # noqa: C408
        title="Feature Value",
        titleside="right",
        tickmode="array",
        tickvals=bar_range,
        ticktext=["Low", "High"],
        ticks="outside",
        thickness=5,
    )


def _add_missing_data_legend(
    fig: go.Figure,
    explanation_for_feature: tp.Dict[str, FeatureShapExplanation],
) -> None:
    contains_nan = [
        feature
        for feature, explanation in explanation_for_feature.items()
        if np.isnan(explanation.data).any()
    ]
    if not contains_nan:
        return

    nan_feature = contains_nan.pop()
    nan_feature_explanation = explanation_for_feature[nan_feature]
    shap_value_for_nan_data = nan_feature_explanation.values[
        np.isnan(nan_feature_explanation.data)
    ]
    nan_feature_index = list(explanation_for_feature.keys()).index(nan_feature)
    fig.add_trace(
        go.Scattergl(
            x=[shap_value_for_nan_data],
            y=[0],
            name="Null",
            mode="markers",
            marker=dict(color="rgb(68,68,68)"),  # noqa: C408
        ),
        row=nan_feature_index + 1,
        col=1,
    )
    fig.update_layout(
        legend=dict(  # noqa: C408
            orientation="h",
            x=1,
            y=0,
            xanchor="left",
            yanchor="top",
        ),
    )


def _get_minmax_normalized_feature(feature_values: np.ndarray) -> np.ndarray:
    min_value = np.nanmin(feature_values)
    max_value = np.nanmax(feature_values)
    return (feature_values - min_value) / (max_value - min_value)


def _jitter_points_based_on_distribution(feature_values: np.ndarray) -> np.ndarray:
    """Returns an array of jittered values that can be used to produce a beeswarm plot
     of the input ``feature_values``.

    This is how the function works:
    - first a distribution of the features g(x) is estimated with KDE
    - then for each feature value x, a random number y is generated using a gaussian
     with average zero and standard deviation g(x)
    The result is that:
    - the y corresponding to x points in areas with many x points are spread of a
     broader range of y values
    - plotting y vs x creates a plot very similar to the shap beeswarm plot
    """
    if len(np.unique(feature_values)) > 1:
        feature_density = stats.gaussian_kde(feature_values).pdf(feature_values)
        jitter_samples = np.random.normal(0, feature_density)
    else:
        jitter_samples = np.zeros_like(feature_values)
    return jitter_samples
