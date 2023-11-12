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
from enum import Enum

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from reporting.charts.utils import apply_chart_style, check_data

CORRELATION_PLOT_COLORBAR_THICKNESS = 20
CORRELATION_PLOT_COLORBAR_TICKS_LENGTH = 3

TStrOrListOfStr = tp.Union[str, tp.List[str]]


class MaskMode(Enum):
    LOWER = "lower"
    UPPER = "upper"


def _get_correlation_data(
    data: pd.DataFrame,
    method: str,
    mask: tp.Optional[MaskMode],
) -> pd.DataFrame:
    """Create correlation matrix.

    Args:
        data: dataframe holding columns' data to calculate correlation
        method: method used to compute correlation. Possible values are:
                * "pearson": standard correlation coefficient.
                * "kendall": Kendall Tau correlation coefficient
                * "spearman": Spearman rank correlation
        mask: mode to mask corr matrix (lower / upper)

    Returns: dataframe holding correlation matrix
    """

    corr_data = data.corr(method=method)

    if mask is None:
        return corr_data

    mask_mode = MaskMode(mask)
    if mask_mode is MaskMode.LOWER:
        corr_data = corr_data.where(np.tri(corr_data.shape[0], k=0).astype(bool))
    elif mask_mode is MaskMode.UPPER:
        corr_data = corr_data.where(np.tri(corr_data.shape[0], k=0).astype(bool)).T

    return corr_data


# Ok to ignore WPS211 for plot functions with many plot options
def plot_correlation(  # noqa: WPS211
    data: pd.DataFrame,
    rows: tp.Optional[tp.List[str]] = None,
    columns: tp.Optional[tp.List[str]] = None,
    sort_by: tp.Optional[TStrOrListOfStr] = None,
    method: str = "pearson",
    mask: tp.Optional[str] = None,
    show_color_bar: bool = True,
    hover_precision: int = 2,
    title: str = "Correlation Plot",
    xaxis_title: tp.Optional[str] = None,
    yaxis_title: tp.Optional[str] = None,
    height: tp.Optional[int] = 700,
    width: tp.Optional[int] = 600,
    fig_params: tp.Optional[tp.Dict[str, tp.Any]] = None,
    layout_params: tp.Optional[tp.Dict[str, tp.Any]] = None,
) -> go.Figure:
    """Create correlation chart

    Args:
        data: dataframe holding correlation matrix
        columns: columns (x-axis) features to use from ``data``
        rows: rows (y-axis) features to use from ``data``
        sort_by: column or set of columns to sort corr matrix by
        method: method used to compute correlation. Possible values are:
                * "pearson": standard correlation coefficient.
                * "kendall": Kendall Tau correlation coefficient
                * "spearman": Spearman rank correlation
        mask: mode to mask corr matrix (lower / upper / None)
        show_color_bar: shows bar scale on the chart if set `true`
        title: title for the chart
        hover_precision: precision of correlation value shown on hover
        xaxis_title: x-axis title for the chart
        yaxis_title: y-axis title for the chart
        height: the height of the exported image in layout pixels
        width: the width of the exported image in layout pixels
        fig_params: kwargs for plotly chart function
        layout_params: dictionary containing keys and values for updating
            the plotly fig layout

    Returns: plotly correlation chart
    """

    data = _filter_input(data, columns, rows)

    corr_map = _get_correlation_data(data, method, mask)
    if columns is not None:
        corr_map = corr_map.loc[:, columns]
    if rows is not None:
        corr_map = corr_map.loc[rows, :]
    if sort_by is not None:
        check_data(corr_map, sort_by)
        corr_map = corr_map.sort_values(sort_by, ascending=False)

    corr_map_fig = go.Heatmap(
        z=corr_map,
        zmin=-1,
        zmid=0,
        zmax=1,
        x=corr_map.columns,
        y=corr_map.index,
        colorscale="RdBu_r",
        colorbar_thickness=CORRELATION_PLOT_COLORBAR_THICKNESS,
        colorbar_ticklen=CORRELATION_PLOT_COLORBAR_TICKS_LENGTH,
        hovertemplate=(
            # String concatenation instead of f-string used to avoid issues
            # when the string has curly braces inside
            "%{x}<br>%{y}<br>%{z:."
            + str(hover_precision)
            + "f}<extra></extra>"
        ),
    )
    fig = go.Figure(data=corr_map_fig, layout=go.Layout(plot_bgcolor="rgba(0,0,0,0)"))

    apply_chart_style(
        fig=fig,
        height=height,
        width=width,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange="reversed",
        fig_params=fig_params,
        layout_params=layout_params,
    )

    if not show_color_bar:
        fig.update_traces(showscale=False)
    return fig


def _filter_input(
    data: pd.DataFrame,
    columns: tp.List[str],
    rows: tp.List[str],
) -> pd.DataFrame:
    if columns is not None and rows is not None:
        features = [*columns, *rows]
        check_data(data, features)
        data = data[features]
    elif columns is not None:
        check_data(data, columns)
    elif rows is not None:
        check_data(data, rows)
    return data
