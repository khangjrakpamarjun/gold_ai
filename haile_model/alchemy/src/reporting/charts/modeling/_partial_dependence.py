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
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import inspection
from sklearn.pipeline import Pipeline

from reporting.api.types import Estimator, Model, SklearnModel

logger = logging.getLogger(__name__)


SUBPLOT_ADJUST_LEFT = 0.125  # the left side of the subplots of the figure
SUBPLOT_ADJUST_RIGHT = 0.9  # the right side of the subplots of the figure
SUBPLOT_ADJUST_BOTTOM = 0.1  # the bottom of the subplots of the figure
SUBPLOT_ADJUST_TOP = 0.9  # the top of the subplots of the figure
# Amount of blank space between subplots
SUBPLOT_ADJUST_WSPACE = 0.2  # the amount of width reserved
SUBPLOT_ADJUST_HSPACE = 0.4  # the amount of height reserved


class ModelThatDoesTransformation(Model, tp.Protocol):
    def transform(
        self,
        data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> tp.Union[pd.DataFrame, np.ndarray, tp.List[tp.Any]]:
        """A methods that transforms data"""


def plot_partial_dependency_for_sklearn(
    data: pd.DataFrame,
    model: tp.Union[Model, SklearnModel],
    features: tp.Optional[tp.List[str]] = None,
    plot_individual_lines: bool = True,
    yaxis_title: str = None,
    title: tp.Optional[str] = "Partial Dependency Plot",
    random_state: int = 42,
    n_columns: int = 2,
    per_feature_height: float = 4,
    plot_width: int = 10,
    n_jobs: int = 1,
    drop_missing_values: bool = True,
) -> plt.Figure:
    """
    Generate a partial dependency plot

    Args:
        data: the dataframe to generate the pdp
        model: base estimator to use for pdp
        features: list of features to build pdp for
        yaxis_title: a title used as a `y` axis label
        title: a title used as a figure sup-title
        plot_individual_lines: plots individual lines if set true;
            plots average lines otherwise
        random_state: used by PartialDependenceDisplay
        n_columns: number of columns in pdp plot
        per_feature_height: size of each feature's plot
        plot_width: pdp plot width
        n_jobs: number of parallel processes to run when estimating pdp
        drop_missing_values: if True, then drops rows with missing values
            in the input data; otherwise features with missing values are not displayed
            which is a bug on sklearn side

    Returns: matplotlib based partial dependency plot
    """
    # todo: filter and order plots using `features`
    if getattr(model, "transform", None) is not None:
        data = _apply_transformation(data, model, drop_missing_values)
    plt.close("all")
    fig = _get_pdp_figure(
        data[model.features_out],
        model.estimator,
        model.features_out,
        n_columns,
        random_state,
        n_jobs,
        plot_individual_lines,
    )  # todo: change uninformative bars on x-axis into nice hist plots
    n_rows = math.ceil(len(model.features_out) / n_columns)
    plot_height = per_feature_height * n_rows
    _apply_styles(fig, title, yaxis_title, plot_height, plot_width, n_columns)
    return fig


def _apply_transformation(
    data: pd.DataFrame,
    model: ModelThatDoesTransformation,
    drop_missing_values: bool,
) -> pd.DataFrame:
    transformed_data = pd.DataFrame(model.transform(data), columns=model.features_out)
    if drop_missing_values:
        transformed_data = _drop_rows_containing_nan(
            transformed_data,
            model.features_out,
        )
    _validate_data(model, transformed_data)
    return transformed_data


def _drop_rows_containing_nan(
    data: pd.DataFrame,
    features: tp.List[str],
) -> pd.DataFrame:
    is_null_row = data[features].isnull().any(axis=1)
    if is_null_row.any():
        n_rows_dropped = is_null_row.sum()
        n_rows_initially = data.shape[0]
        logger.warning(
            f"Got nan values in the provided features. "
            f"Sklearn pdp doesn't know how to deal with those. "
            f"Such rows will be dropped. "
            f"Amount of rows dropped out of all for the visual: "
            f"{n_rows_dropped}/{n_rows_initially}",
        )
    return data.dropna(subset=features)


def _get_pdp_figure(
    data: pd.DataFrame,
    estimator: Estimator,
    features: tp.List[str],
    n_columns: int,
    random_state: int,
    n_jobs: int,
    plot_sampled_dependency_lines: bool,
) -> plt.Figure:
    try:
        pdp_plotting_function = inspection.PartialDependenceDisplay.from_estimator
    except AttributeError:
        # If we got here, this means that sklearn version is <1.0.0.
        # For this version `Pipeline` implementation has a bug.
        if isinstance(estimator, Pipeline):
            _mark_as_fitted(estimator)
        pdp_plotting_function = inspection.plot_partial_dependence
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        fig = pdp_plotting_function(
            estimator=estimator,
            X=data,
            features=features,
            random_state=random_state,
            n_cols=n_columns,
            n_jobs=n_jobs,
            kind="both" if plot_sampled_dependency_lines else "average",
        ).figure_
    return fig


def _mark_as_fitted(pipeline: Pipeline) -> None:
    """
    Adds dummy attribute to pipeline steps that help to pass sklearn `check_is_fitted`

    Notes:
        If you call `sklearn.utils.validation.check_is_fitted`
        on a fitted Pipeline instance it will raise `NotFittedError`.
        To support backward compatibility for earlier sklearn version
        we are overcoming this bug by adding an attribute ending _ to each step,
        which by sklearn convention states that estimator is fitted
        allowing the sklearn `check_is_fitted` method to pass.
    """
    for _, step in pipeline.steps:
        step.is_fitted_ = True


def _turn_off_spine_ticks(ax: plt.Axes):
    """Turn off the right and top spine/ticks"""
    for position in ("right", "top"):
        ax.spines[position].set_color("none")
        ax.yaxis.tick_left()


def _identify_major_formatter(ax, axis_index, n_columns, current_major_formatter):
    """Identifies which major formatter has to be used in the loop.

    - if the current `ax` is the first and main axis in the row, it sets the
    major y-axis formatter from that axes as the new `current_major_formatter`
    - otherwise keeps the current `current_major_formatter`
    """
    is_first_axis_in_row = not bool(
        axis_index % n_columns,
    )  # first (and main) axis in row
    if is_first_axis_in_row:
        return ax.yaxis.get_major_formatter()
    return current_major_formatter


def _apply_style_to_axis(
    ax: plt.Axes,
    major_formatter,
    show_grid: bool,
    average_line_color,
    individual_lines_color,
):
    """Applies styles to one axis

    Modifies `ax` in place.
    """
    # update x and y labels
    ax.set_xlabel(ax.get_xlabel(), loc="right", weight="bold")
    ax.set_ylabel(ax.get_ylabel(), weight="bold", labelpad=10)

    # (Because of how this is used in the row,
    #   the next line of code has the effect of showing the ticks everywhere)
    # show ticks everywhere
    ax.yaxis.set_major_formatter(major_formatter)

    #  turn off the right and top spine/ticks
    _turn_off_spine_ticks(ax)

    # add dotted grid
    if show_grid:
        ax.grid(visible=True, which="both", linestyle="--")

    # change lines colors
    for line in ax.get_lines():
        if line.get_label() == "average":
            line.set(color=average_line_color, linewidth=3)
        else:
            line.set_color(individual_lines_color)
    # fix legend color
    legend = ax.get_legend()
    if legend:
        # there's only 1 legend with our average line
        legend.legendHandles[0].set_color(average_line_color)


def _apply_styles(
    fig: plt.Figure,
    title: tp.Optional[str],
    yaxis_title: tp.Optional[str],
    plot_height: float,
    plot_width: float,
    n_columns: int,
    title_position: float = 0.95,
    title_font_size: int = 23,
    show_grid: bool = True,
    average_line_color: str = "#09346d",
    individual_lines_color: str = "#609ef1",
) -> None:
    plt.subplots_adjust(
        left=SUBPLOT_ADJUST_LEFT,
        right=SUBPLOT_ADJUST_RIGHT,
        bottom=SUBPLOT_ADJUST_BOTTOM,
        top=SUBPLOT_ADJUST_TOP,
        wspace=SUBPLOT_ADJUST_WSPACE,
        hspace=SUBPLOT_ADJUST_HSPACE,
    )

    fig.set_size_inches(plot_width, plot_height)
    if title:
        fig.suptitle(title, fontsize=title_font_size, y=title_position)

    if yaxis_title is not None:
        labeled_axes = [axis for axis in fig.get_axes() if axis.get_ylabel()]
        for axis in labeled_axes:
            axis.set_ylabel(yaxis_title)

    major_formatter = None
    ax: plt.Axes
    # for some reason sklearn creates empty 0th axis
    non_empty_axes = fig.axes[1:]
    for axis_index, ax in enumerate(non_empty_axes):
        major_formatter = _identify_major_formatter(
            ax,
            axis_index,
            n_columns,
            current_major_formatter=major_formatter,
        )
        _apply_style_to_axis(
            ax,
            major_formatter,
            show_grid,
            average_line_color,
            individual_lines_color,
        )


def _validate_data(model: Model, transformed_data: pd.DataFrame) -> None:
    if any(transformed_data.isnull().any()):
        raise ValueError(
            "Unfortunately sklearn pdp doesn't work with nan values",  # todo: check
        )
    if transformed_data.columns.tolist() != model.features_out:
        raise ValueError(
            "Invariant violated. Expected data columns to be equal to "
            "`model.features_out` after the `.transform`",
        )
