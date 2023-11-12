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
Convergence plots.
"""

import matplotlib.pyplot as plt
import numpy as np

from optimizer.loggers import BasicLogger
from optimizer.types import Vector


def convergence_plot(
    overall_min_objective: Vector = None,
    overall_max_objective: Vector = None,
    mean_objective: Vector = None,
    std_objective: Vector = None,
    logger: BasicLogger = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """Simple plot of the convergence of given values.
    Two plotting modes depending on the arguments given:

    Logger:
        Takes min, max, mean, and standard deviations of objectives from
        the BasicLogger object and plots them.
    Provided values:
        Takes the provided min, max, mean, and standard deviations and plots the
        values that are provided. If one is not given, it will not be plotted.
    Both:
        Provided value will override

    Args:
        overall_min_objective: Vector of minimum objective values.
        overall_max_objective: Vector of maximum objective values.
        mean_objective: Vector of mean objective values.
        std_objective: Vector of objective values standard deviations.
        logger: BasicLogger, the above objectives will be taken from here.
        ax: Axes, optional axes.

    Returns:
        plt.Axes, Axes objects of the plot.

    Raises:
        ValueError: if no values or BasicLogger are provided.
    """
    objectives = [
        overall_min_objective,
        overall_max_objective,
        mean_objective,
        std_objective,
    ]

    if all(value is None for value in objectives + [logger]):
        raise ValueError(
            "No values provided to plot. "
            "Provide objective value Vectors or a BasicLogger."
        )

    if logger is not None:
        df = logger.log_df

        overall_min_objective = (
            overall_min_objective
            if overall_min_objective is not None
            else df["overall_min_objective"]
        )

        overall_max_objective = (
            overall_max_objective
            if overall_max_objective is not None
            else df["overall_max_objective"]
        )

        mean_objective = (
            mean_objective if mean_objective is not None else df["mean_objective"]
        )

        std_objective = (
            std_objective if std_objective is not None else df["std_objective"]
        )

    ax = ax or plt.gca()

    if overall_min_objective is not None:
        ax.plot(
            overall_min_objective,
            color="C0",
            label="Minimum Objective",
        )

    if overall_max_objective is not None:
        ax.plot(
            overall_max_objective,
            color="C1",
            label="Maximum Objective",
        )

    if mean_objective is not None:
        ax.plot(mean_objective, color="C2", label="Mean Objective")

        if std_objective is not None:
            ax.fill_between(
                np.arange(len(mean_objective)),
                mean_objective + std_objective,
                mean_objective - std_objective,
                alpha=0.2,
                color="C2",
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Value")
    ax.legend()

    return ax
