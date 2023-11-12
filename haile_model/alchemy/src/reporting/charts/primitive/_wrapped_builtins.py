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
import plotly.express as px
import plotly.graph_objects as go

from reporting.charts.utils import apply_chart_style, check_data
from reporting.config import COLORS

TDict = tp.Dict[str, tp.Any]
TStrOrListOfStr = tp.Union[str, tp.List[str]]


# Ok to ignore WPS211 for plot functions with many plot options
def plot_bar(  # noqa: WPS211
    data: pd.DataFrame,
    x: tp.Optional[str] = None,  # noqa: WPS111
    y: tp.Optional[str] = None,  # noqa: WPS111
    color: tp.Optional[str] = None,
    color_discrete_map: tp.Optional[tp.List[str]] = None,
    sort_by: tp.Union[str, tp.List[str]] = None,
    orientation: str = "h",
    barmode: tp.Optional[str] = None,
    title: str = "Bar Chart",
    xaxis_title: tp.Optional[str] = None,
    yaxis_title: tp.Optional[str] = None,
    height: tp.Optional[int] = None,
    width: tp.Optional[int] = None,
    fig_params: tp.Optional[TDict] = None,
    layout_params: tp.Optional[TDict] = None,
) -> go.Figure:
    """Create bar chart.

    Args:
        data: dataframe for chart
        x: column name of column holding x axis data dataframe
        y: column name of column holding y axis data dataframe
        color: column name of column holding hue/color data dataframe
        color_discrete_map: list of colors to use when coloring by `color` column
        sort_by: name or list of names of columns to sort by
        orientation: sets to "h" when plotting horizontal bar
        barmode: Determines how bars at the same location coordinate are
                  displayed on the graph.
        title: title for the chart
        xaxis_title: x-axis title for the chart
        yaxis_title: y-axis title for the chart
        height: the height of the exported image in layout pixels
        width: the width of the exported image in layout pixels
        fig_params: kwargs for plotly chart function
        layout_params: dictionary containing keys and values for updating
         the plotly fig layout

    Returns: plotly bar chart
    """

    if sort_by is not None:
        data = data.sort_values(by=sort_by, ascending=True)

    return _build_figure(
        graph_object_producer=px.bar,
        graph_object_producer_kwargs=dict(  # noqa: C408
            orientation=orientation,
            barmode=barmode,
        ),
        data=data,
        x=x,
        y=y,
        color=color,
        color_discrete_map=color_discrete_map,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        width=width,
        fig_params=fig_params,
        layout_params=layout_params,
    )


# Ok to ignore WPS211 for plot functions with many plot options
def plot_box(  # noqa: WPS211
    data: pd.DataFrame,
    x: tp.Optional[str] = None,  # noqa: WPS111
    y: tp.Optional[str] = None,  # noqa: WPS111
    color: tp.Optional[str] = None,
    color_discrete_map: tp.Optional[tp.List[str]] = None,
    quartilemethod: tp.Optional[str] = "linear",
    title: str = "Bar Chart",
    xaxis_title: tp.Optional[str] = None,
    yaxis_title: tp.Optional[str] = None,
    height: tp.Optional[int] = None,
    width: tp.Optional[int] = None,
    fig_params: tp.Optional[TDict] = None,
    layout_params: tp.Optional[TDict] = None,
) -> go.Figure:
    """Create box chart.

    Args:
        data: dataframe for chart
        x: column name of column holding x axis data dataframe
        y: column name of column holding y axis data dataframe
        color: column name of column holding hue/color data dataframe
        color_discrete_map: list of colors to use when coloring by `color` column
        quartilemethod: sets the method used to compute the sample's Q1 and Q3
        title: title for the chart
        xaxis_title: x-axis title for the chart
        yaxis_title: y-axis title for the chart
        height: the height of the exported image in layout pixels
        width: the width of the exported image in layout pixels
        fig_params: kwargs for plotly chart function
        layout_params: dictionary containing keys and values for updating
         the plotly fig layout

    Returns: plotly box chart
    """

    fig = _build_figure(
        graph_object_producer=px.box,
        graph_object_producer_kwargs={},
        data=data,
        x=x,
        y=y,
        color=color,
        color_discrete_map=color_discrete_map,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        width=width,
        fig_params=fig_params,
        layout_params=layout_params,
    )
    fig.update_traces(quartilemethod=quartilemethod)

    return fig


# Ok to ignore WPS211 for plot functions with many plot options
def plot_histogram(  # noqa: WPS211
    data: pd.DataFrame,
    x: tp.Optional[str] = None,  # noqa: WPS111
    y: tp.Optional[str] = None,  # noqa: WPS111
    histnorm: tp.Optional[str] = None,
    nbins: tp.Optional[int] = None,
    barmode: tp.Optional[str] = "overlay",
    color: tp.Optional[str] = None,
    color_discrete_map: tp.Optional[tp.List[str]] = None,
    title: str = "Histogram Chart",
    xaxis_title: tp.Optional[str] = None,
    yaxis_title: tp.Optional[str] = None,
    height: tp.Optional[int] = None,
    width: tp.Optional[int] = None,
    fig_params: tp.Optional[TDict] = None,
    layout_params: tp.Optional[TDict] = None,
) -> go.Figure:
    """Create histogram chart.

    Args:
        data: dataframe for chart
        x: column name of column holding x-axis data dataframe
        y: column name of column holding y-axis data dataframe
            if not specified, y-axis represents the frequency of the data
        histnorm: sets the method used to calculate frequency. Available
                options are "probability", "percent", "density", "probability density"
        nbins: sets the number of bins
        barmode: sets `barmode` for `plotly.express.histogram`
        color: column name of column holding hue/color data dataframe
        color_discrete_map: list of colors to use when coloring by `color` column
        title: title for the chart
        xaxis_title: x-axis title for the chart
        yaxis_title: y-axis title for the chart
        height: the height of the exported image in layout pixels
        width: the width of the exported image in layout pixels
        fig_params: kwargs for plotly chart function
        layout_params: dictionary containing keys and values for updating
         the plotly fig layout

    Returns: plotly histogram chart
    """
    return _build_figure(
        graph_object_producer=px.histogram,
        graph_object_producer_kwargs=dict(  # noqa: C408
            histnorm=histnorm,
            nbins=nbins,
            barmode=barmode,
        ),
        data=data,
        x=x,
        y=y,
        color=color,
        color_discrete_map=color_discrete_map,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        width=width,
        fig_params=fig_params,
        layout_params=layout_params,
    )


# Ok to ignore WPS211 for plot functions with many plot options
def plot_scatter(  # noqa: WPS211
    data: pd.DataFrame,
    x: tp.Optional[str] = None,  # noqa: WPS111
    y: tp.Optional[str] = None,  # noqa: WPS111
    color: tp.Optional[str] = None,
    color_discrete_map: tp.Optional[tp.List[str]] = None,
    trendline: tp.Optional[str] = None,
    title: str = "Scatter Plot",
    xaxis_title: tp.Optional[str] = None,
    yaxis_title: tp.Optional[str] = None,
    height: tp.Optional[int] = None,
    width: tp.Optional[int] = None,
    fig_params: tp.Optional[TDict] = None,
    layout_params: tp.Optional[TDict] = None,
) -> go.Figure:
    """Create scatter chart.

    Args:
        data: dataframe for chart
        x: column name of column holding x axis data dataframe
        y: column name of column holding y axis data dataframe
        color: column name of column holding hue/color data dataframe
        color_discrete_map: list of colors to use when coloring by `color` column
        trendline: sets the method to add regression trendline to scatterplot.
                    option "ols" fits an ordinary least square regression
        title: title for the chart
        xaxis_title: x-axis title for the chart
        yaxis_title: y-axis title for the chart
        height: the height of the exported image in layout pixels
        width: the width of the exported image in layout pixels
        fig_params: kwargs for plotly chart function
        layout_params: dictionary containing keys and values for updating
         the plotly fig layout

    Returns: plotly scatter chart
    """
    return _build_figure(
        graph_object_producer=px.scatter,
        graph_object_producer_kwargs=dict(trendline=trendline),  # noqa: C408
        data=data,
        x=x,
        y=y,
        color=color,
        color_discrete_map=color_discrete_map,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        width=width,
        fig_params=fig_params,
        layout_params=layout_params,
    )


def plot_timeline(
    data: pd.DataFrame,
    x: tp.Optional[str] = None,  # noqa: WPS111
    y: tp.Optional[TStrOrListOfStr] = None,  # noqa: WPS111
    color: str = None,
    color_discrete_map: tp.Optional[tp.List[str]] = None,
    title: str = "Timeline Chart",
    xaxis_title: tp.Optional[str] = "Time",
    yaxis_title: tp.Optional[str] = None,
    height: tp.Optional[int] = None,
    width: tp.Optional[int] = None,
    fig_params: tp.Optional[TDict] = None,
    layout_params: tp.Optional[TDict] = None,
) -> go.Figure:
    """Create time series chart

    Args:
        data: dataframe for chart
        x: column name of column holding timestamp data
        y: column name of column holding y axis data dataframe
        color: column name of column holding hue/color data dataframe
        color_discrete_map: list of colors to use when coloring by `color` column
        title: title for the chart
        xaxis_title: x-axis title for the chart
        yaxis_title: y-axis title for the chart
        height: the height of the exported image in layout pixels
        width: the width of the exported image in layout pixels
        fig_params: kwargs for plotly chart function
        layout_params: dictionary containing keys and values for updating
         the plotly fig layout

    Returns: plotly time series chart
    """
    return _build_figure(
        graph_object_producer=px.line,
        graph_object_producer_kwargs={},
        data=data,
        x=x,
        y=y,
        color=color,
        color_discrete_map=color_discrete_map,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        width=width,
        fig_params=fig_params,
        layout_params=layout_params,
    )


class _FigureGenerator(tp.Protocol):
    def __call__(
        self,
        data_frame: pd.DataFrame,
        x: tp.Optional[str],  # noqa: WPS111
        y: tp.Optional[str],  # noqa: WPS111
        color: tp.Optional[str],
        color_discrete_sequence: tp.List[str],
        *args: tp.Any,
        **kwargs: tp.Any,
    ) -> go.Figure:
        """A figure generator shoild return a go.Figure when called"""


# Ok to ignore WPS211 for plot functions with many plot options
def _build_figure(  # noqa: WPS211
    graph_object_producer: _FigureGenerator,
    graph_object_producer_kwargs: TDict,
    data: pd.DataFrame,
    x: tp.Optional[str],  # noqa: WPS111
    y: tp.Optional[str],  # noqa: WPS111
    color: tp.Optional[str],
    color_discrete_map: tp.Optional[tp.List[str]],
    title: str,
    xaxis_title: tp.Optional[str],
    yaxis_title: tp.Optional[str],
    height: tp.Optional[int],
    width: tp.Optional[int],
    fig_params: tp.Optional[TDict],
    layout_params: tp.Optional[TDict],
) -> go.Figure:
    check_data(data, x, y)
    fig = graph_object_producer(
        data_frame=data,
        x=x,
        y=y,
        color=color,
        color_discrete_sequence=color_discrete_map if color_discrete_map else COLORS,
        **graph_object_producer_kwargs,
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
    return fig
