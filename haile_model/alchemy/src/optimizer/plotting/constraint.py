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
Constraint related plots.
"""

from collections import defaultdict
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from optimizer.loggers import PenaltyLogger


def penalty_plot(
    logger: PenaltyLogger = None,
    ax: plt.Axes = None,
    normalize_penalties: bool = True,
    agg: Callable[[np.ndarray], float] = np.mean,
) -> plt.Axes:
    """Show the calculated penalty values over time.

    Args:
        logger: BasicLogger, the above objectives will be taken from here.
        ax: Axes, optional axes.
        normalize_penalties: bool, True to convert penalty values to 0 to 1.
        agg: aggregation function.

    Returns:
        plt.Axes, Axes objects of the plot.

    """
    data = logger.data_dict
    ax = ax or plt.gca()

    penalty_means = defaultdict(list)

    # Make sure we access each value in ascending order.
    keys = sorted(list(data.keys()))

    for key in keys:
        for penalty_name, info in data[key].items():
            penalties = info["penalty"]
            penalty_means[penalty_name].append(agg(penalties))

    names = sorted(list(penalty_means.keys()))

    for penalty_name in names:
        mean_vals = np.array(penalty_means[penalty_name])

        if normalize_penalties:
            mean_vals = MinMaxScaler().fit_transform(mean_vals[:, np.newaxis])

        ax.plot(mean_vals, label=agg.__name__.title() + " " + penalty_name)

    ax.set_xlabel("Iteration")

    ylabel = "Penalty Value"

    if normalize_penalties:
        ylabel += " (Normalized)"

    ax.set_ylabel(ylabel)
    ax.legend()

    return ax
