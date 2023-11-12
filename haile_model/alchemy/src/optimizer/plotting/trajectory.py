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

"""
Best solution at each iteration plots.
"""
from typing import List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import MinMaxScaler

from optimizer.constraint import InequalityConstraint, Penalty
from optimizer.loggers import BestTrajectoryLogger
from optimizer.problem import StatefulOptimizationProblem
from optimizer.types import Matrix, Vector
from optimizer.utils.diagnostics import get_penalties_table, get_slack_table


def _minmax_bounds_scale(solutions: Matrix, bounds: Matrix) -> Matrix:
    """Scales solution variables using the bounds for each of them. Thus, if
    x = max_bound_x, then it will be 1; and if x = min_bound_x the outcome will be 0.
    If x is outside its bounds, then the result will be outside the range [0,1].

    Args:
        solutions: unscaled solution variables
        bounds: Matrix where row i contains the lower and upper bound for variable
        i (column i in the solution matrix)

    Returns:
        Matrix of shape solutions.shape with variables scaled by using the bounds
        (if no bound is violated, then the range of this matrix is [0,1])

    """
    scaled = (solutions - bounds.min(axis=1)) / (
        bounds.max(axis=1) - bounds.min(axis=1)
    )
    return scaled


def _min_zero_scale(m: Matrix) -> Matrix:
    """Map values of each variable in m (variable=column) from [min to 0] into [0 to 1].
     This works well only if the column of m has at least one negative value. If this
     does not hold, then the output becomes <= 0 for the entire column; and we actually
     never use the output of this function because we filter it in _plot_slack by other
    means. Finally, when some value in m is above 0 (and the column has a negative
    value), then this value is mapped to some point >1.

    This function is used for representing the slacks for different penalties in a
    normalized color range in a heatmap. If the slack <= 0, then it means that there is
    room to increase the constraint and get closer to the bound. We want to capture the
    closeness to 0 with a higher value; while keeping unconstrained the highest distance
    to boundary. That's why we compute this and use the values where the scaled range is
    between 0 and 1, as the range between 'the least violated slack' to 'the almost
    violated slack'; if we didn't have a negative value in some column in m, this means
    that we always violated the boundary and we color the heatmap with some other color.

    Args:
        m: matrix

    Returns:
        Matrix whose columns containing a negative value are scaled between 0 and 1 for
        values in min-0 range, and >1
        for values above 0. If a column contains no negative value, returns a <= 0 value
        for each entry in the column.
    """
    scaled = (m - m.min(axis=0)) / (-m.min(axis=0))
    return scaled


def _plot_slack(
    fig: plt.Figure, ax: plt.Axes, solutions: Matrix, penalties: List[Penalty]
) -> plt.Axes:
    """Plots heatmap with color-coded distances for evaluated constraints to the bounds.
    The darked the blue, the smaller the slack / room the constraint has until reaching
    the bound (i.e., the tighter the constraint). Penalties that are not
    InequalityConstraint based will be silently ommited. The rationale is that we might
    receive a problem that has more than one type of penalties and we don't want to
    alert more than needed.

    Args:
        fig: Figure in which this will be drawn
        ax: Axes on which this will be drawn
        solutions: Solutions for which slack will be computed and plotted
        penalties: Penalties from which InequalityContraints are extracted and evaluated

    Returns:
        plt.Axes with the plot on it. This is the same object as the received by
        parameter.

    """
    inequality_penalties = [
        p for p in penalties if isinstance(p.constraint, InequalityConstraint)
    ]
    if len(inequality_penalties) == 0:
        ax.set_title("Slack Values: No inequality constraints")
        return ax
    # get_slack_table has a convention of positive being "amount left".
    # In plot, slack are "negative" values, so multiply by -1 once more.
    slack_values = -1 * get_slack_table(solutions, inequality_penalties).values
    slack_scaled_values = _min_zero_scale(slack_values)
    im = ax.imshow(slack_scaled_values.T, aspect="auto", cmap="Blues", vmax=1)
    ax.imshow(
        np.where(slack_values.T > 0, 0.7, np.nan),  # .7 is makes a nice red shade
        aspect="auto",
        cmap="Reds",
        vmin=0,
        vmax=1,
    )
    ax.set_title("Slack Values")
    ax.set_yticks(range(slack_values.T.shape[0]))
    ax.set_yticklabels(
        [
            p.name if p.name is not None else f"p_{i}"
            for i, p in enumerate(inequality_penalties)
        ]
    )

    cax = inset_axes(ax, width="1%", height="100%", loc="right", borderpad=-2)
    cbar = fig.colorbar(
        im, cax=cax, ticks=[slack_scaled_values.min(), 1], orientation="vertical"
    )
    cbar.ax.set_yticklabels(["Far from Bound", "Close to Bound"])

    patch = mpatches.Patch(color="red", label="Violated")
    ax.legend(handles=[patch], loc=(0.91, 1.07), borderaxespad=0.0)
    return ax


def _plot_penalties(
    fig: plt.Figure,
    ax: plt.Axes,
    solutions: Matrix,
    penalties: List[Penalty],
    normalize_penalties: bool,
) -> plt.Axes:
    """Plots heatmap with penalties color-coded. Blue means that penalty was not
    violated.
    If a penalty was triggered, then the darker the red the bigger the penalty.
    If normalize_penalties is True, the red shades will be scaled independently for each
    penalty.

    Args:
        fig: plt.Figure on which the plot is displayed
        ax: plt.Axes used for the plot
        solutions: Solutions matrix that will be analysed
        penalties: List of penalties to evaluate on solutions
        normalize_penalties: set to True for making a relative coloring for each of the
        penalties independently. Note that if True, the plot will will have intense reds
        for penalties that might be penalizing little relative to other penalties that
        are actually damaging our objective value.

    Returns:
        plt.Axes with the plot on it. This is the same object as the received as
        parameter

    """
    if len(penalties) == 0:
        ax.set_title("Penalty constraint violations: No penalties available")
        return ax
    penalty_values = get_penalties_table(solutions, penalties).values
    if normalize_penalties:
        penalty_values = MinMaxScaler().fit_transform(penalty_values)
    im = ax.imshow(penalty_values.T, aspect="auto", cmap="Reds")
    ax.imshow(
        np.where(
            penalty_values.T == 0, 0.7, np.nan
        ),  # the .7 makes a nice shade of blue
        aspect="auto",
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    ax.set_title("Penalty constraint violations")
    ax.set_yticks(range(penalty_values.T.shape[0]))
    ax.set_yticklabels(
        [p.name if p.name is not None else f"p_{i}" for i, p in enumerate(penalties)]
    )

    cax = inset_axes(ax, width="1%", height="100%", loc="right", borderpad=-2)
    cbar = fig.colorbar(
        im,
        cax=cax,
        ticks=[penalty_values.min(), penalty_values.max()],
        orientation="vertical",
    )
    cbar.ax.set_yticklabels(["Low", "High"])

    patch = mpatches.Patch(color="tab:blue", label="Non Violated")
    ax.legend(handles=[patch], loc=(0.91, 1.07), borderaxespad=0.0)
    return ax


def _plot_convergence(ax: plt.Axes, objective_values: Vector) -> plt.Axes:
    """Plots a time series with the objective values.

    Args:
        ax: Axes on which the time series will be plotted
        objective_values: Objective function values

    Returns:
        plt.Axes with the plot on it. This is the same object as the received as
        parameter

    """
    ax.plot(objective_values, drawstyle="steps-post")
    ax.set_title("Convergence Plot")
    ax.set_ylabel("Objective Value")
    return ax


def _plot_solutions(
    fig: plt.Figure,
    ax: plt.Axes,
    solutions: Matrix,
    bounds: Matrix,
    variable_labels: List = None,
) -> plt.Axes:
    """Plots heatmap with variable values color coded: the darker the color, the closer
    the value to it's upper bound.
    If the value for some variable is out of bounds, then it will be colored by red or
    blue in the heatmap.

    Args:
        fig: Figure on which this will be plotted
        ax: Axes on which this will be plotted
        solutions: Matrix where each row is a solution to be plotted
        bounds: Matrix where each row i contains the bounds for variable i (ith column
        on the solutions matrix)
        variable_labels: Labels to display on the left of the heatmap. If not provided,
        the label will be the index

    Returns:
        plt.Axes with the plot on it. This is the same object as the received as
        parameter

    """
    y = _minmax_bounds_scale(solutions, bounds)
    im = ax.imshow(y.T, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.imshow(
        np.where(y.T > 1, 0.7, np.nan),
        aspect="auto",
        cmap="Reds",
        vmin=0,
        vmax=1,  # The .7 makes a nice shade of red
    )
    ax.imshow(
        np.where(y.T < 0, 0.7, np.nan),
        aspect="auto",
        cmap="Blues",
        vmin=0,
        vmax=1,  # The .7 makes a nice shade of blue
    )
    ax.set_title("Optimized Variables")
    ax.set_yticks(range(y.T.shape[0]))
    if variable_labels:
        ax.set_yticklabels(variable_labels)

    cax = inset_axes(ax, width="1%", height="100%", loc="right", borderpad=-2)
    cbar = fig.colorbar(im, cax=cax, ticks=[0, 1], orientation="vertical")
    cbar.ax.set_yticklabels(["Lower Bound", "Higher Bound"])
    return ax


def _estimate_plot_size(
    solutions: Matrix,
    penalties: List[Penalty],
    height_scale: float = 0.4,
    time_series_size: float = 7,
) -> Tuple[float, List[float]]:
    """Estimate the total height and the ratios of the axes for plotting. It parses the
    input and does the estimation based on the amount of variables, penalties and the
    amount of penalties with slack. If you wish to increment/reduce the plot size, pass
    a different height_scale factor.

    Args:
        solutions: matrix containing the solutions
        penalties: list of penalties
        height_scale: multiplier of the height
        time_series_size: factor that indicates how many heatmap row heights contains
        the time series plot (approx.) Default value is 7, that it, the time series
        height is equivalent to 7 heatmap rows.

    Returns:
        A tuple (total_height, height_ratios), where total_height is an estimation of
        the figure total height, while the height_ratios is a list that's useful as
        height_ratios entry in matplotlib's gridspec_kw dictionary. The ratios are in
        the following order: [convergence time series, variables heatmap,
        slack heatmap, penalties heatmap]
    """
    sizes = [
        time_series_size,
        max(solutions.shape[1], 2),
        max(
            len(
                [p for p in penalties if isinstance(p.constraint, InequalityConstraint)]
            ),
            2,
        ),
        max(len(penalties), 2),
    ]
    return sum(sizes) * height_scale, sizes


def best_trajectory_problem_plot(
    best_trajectory_logger: BestTrajectoryLogger,
    problem: StatefulOptimizationProblem,
    bounds: List[Tuple],
    normalize_penalties: bool = False,
) -> (plt.Figure, List[plt.Axes]):
    """Generates a plot on how the optimization was executed, containing 4 charts:

        - The first chart is the objective value as function of the iteration in logger
        - The second captures how far from its bounds were the optimizable varibles in
          a heatmap. The darker the shade, the closer the value to the upper bound.
        - The third captures the distance of each constraint to it's bound (inequality
          penalty). The darker the shade, the closer the constraint was to the bound
          –– in other words, the tighter the constraint. If the bound is violated, then
          the occurrence is colored in red.
        - The fourth captures the impact of the penalties on the objective value. The
          darker the shade, the more this penalty contributed to the objective value.
          If the penalty was not triggered (value = 0), then the occurrence is
          colored in blue.

    Args:
        best_trajectory_logger: The logger that logged during the optimization
            execution.
        problem: A StatefulOptimizationProblem. This plotting function will use the
            problem penalties to build the plots.
        bounds: A list of (min_bound, max_bound) for each of the variables. This has to
            be in the same order as the solutions columns.
        normalize_penalties: If True, this will make a normalization for each Penalty.
            This might be useful if at some iteration one of the penalties happens to
            be very large and thus the shades in the heatmap towards the end of the
            run get very light. To read it better, it could be interesting to normalize
            these svalues.

    Returns:
        A tuple (plt.Figure, List[plt.Axes]) with the plots.
    """
    return _best_trajectory_plot(
        best_trajectory_logger.solutions,
        problem.substitute_parameters(best_trajectory_logger.solutions),
        best_trajectory_logger.objective_values,
        bounds,
        problem.penalties,
        variable_labels=problem.optimizable_columns,
        normalize_penalties=normalize_penalties,
    )


def best_trajectory_plot(
    best_trajectory_logger: BestTrajectoryLogger,
    penalties: List[Penalty],
    bounds: List[Tuple],
    normalize_penalties: bool = False,
    variable_labels: List[str] = None,
) -> (plt.Figure, List[plt.Axes]):
    """
    Generates a plot on how the optimization was executed, containing 4 charts:

        - The first chart is the objective value as function of the iteration in logger
        - The second captures how far from its bounds were the optimizable varibles
          in a heatmap. The darker the shade, the closer the value to the upper bound.
        - The third captures the distance of each constraint to it's bound (inequality
          penalty). The darker the shade, the closer the constraint was to the bound
          –– in other words, the tighter the constraint. If the bound is violated,
          then the occurrence is colored in red.
        - The fourth captures the impact of the penalties on the objective value. The
          darker the shade, the more this penalty contributed to the objective value.
          If the penalty was not triggered (value = 0), then the occurrence is colored
          in blue.

    Args:
        best_trajectory_logger: The logger that logged during the optimization
            execution.
        penalties: A list of Penalty objects used for slack and penalty plots.
        bounds: A list of (min_bound, max_bound) for each of the variables. This has to
            be in the same order as the solutions columns.
        normalize_penalties: If True, this will make a normalization for each Penalty.
            This might be useful if at some iteration one of the penalties happens to
            be very large and thus the shades in the heatmap towards the end of the
            run get very light. To read it better, it could be interesting to normalize
            these values.
        variable_labels: labels list for adding them on the y axis of the variable vs.
            their bounds plot.

    Returns:
        A tuple (plt.Figure, List[plt.Axes]) with the plots.
    """
    return _best_trajectory_plot(
        best_trajectory_logger.solutions,
        best_trajectory_logger.solutions,
        best_trajectory_logger.objective_values,
        bounds,
        penalties,
        normalize_penalties,
        variable_labels=variable_labels,
    )


def _best_trajectory_plot(
    solutions: Matrix,
    contextual_solutions: Matrix,
    objective_values: Vector,
    bounds: List[Tuple],
    penalties: List[Penalty],
    normalize_penalties: bool = False,
    variable_labels: List[str] = None,
) -> (plt.Figure, List[plt.Axes]):
    """Generates a plot on how the optimization was executed, containing 4 charts:
        - The first chart is the objective value as function of the iteration in logger
        - The second captures how far from its bounds were the optimizable varibles in a
        heatmap. The darker the shade, the closer the value to the upper bound.
        - The third captures the distance of each constraint to it's bound (inequality
        penalty). The darker the shade, the closer the constraint was to the bound –– in
         other words, the tighter the constraint. If the bound is violated, then the
         occurrence is colored in red.
        - The fourth captures the impact of the penalties on the objective value. The
        darker the shade, the more this penalty contributed to the objective value. If
        the penalty was not triggered (value = 0), then the occurrence is colored in
        blue.

    Args:
        solutions: Solutions that will be plotted in the optimized variables heatmap
        (2nd plot)
        contextual_solutions: Solutions that will be used to evaluate the panalties and
        slacks. Note that contextual_solutions and solutions can be the same if the
        penalties are based only on the optimized variables and are not labeled.
        objective_values: Values that will appear in the convergence plot
        penalties: This can be either a list of Penalty objects, or a
        StatefulOptimizationProblem. In the latter case, this plotting function will use
         the problem penalties to build the plots.
        - bounds: A list of (min_bound, max_bound) for each of the variables. This has
        to be in the same order as the solutions columns.
        normalize_penalties: If True, this will make a normalization for each Penalty.
        This might be useful if at some iteration one of the penalties happens to be
        very large and thus the shades in the heatmap towards the end of the run get
        very light. To read it better, it could be interesting to normalize these
        values.
        variable_labels: labels list for adding them on the y axis of the variable vs.
        their bounds plot.

    Returns:
        A tuple (plt.Figure, List[plt.Axes]) with the plots.
    """

    height, height_ratios = _estimate_plot_size(solutions, penalties)

    fig, ax = plt.subplots(
        figsize=(15, height),
        nrows=4,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )

    _plot_convergence(ax[0], objective_values)
    _plot_solutions(
        fig, ax[1], solutions, np.stack(bounds), variable_labels=variable_labels
    )
    _plot_slack(fig, ax[2], contextual_solutions, penalties)
    _plot_penalties(fig, ax[3], contextual_solutions, penalties, normalize_penalties)

    ax[-1].set_xlabel("Iteration")

    return fig, ax
