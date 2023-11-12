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
from numbers import Real
from typing import Tuple

import numpy as np

from optimizer.types import Vector


def get_best_minimize(
    objective_values: Vector, current_min: Real
) -> Tuple[Real, float]:
    """Get the best from a list of objectives and the current best value.

    Args:
        objective_values: Vector of objectives.
        current_min: current minimum objective value.

    Returns:
        (Real, float), minimum found value and the current minimum minus this value.
    """
    objectives_best = np.min(objective_values)

    overall_best = np.min([objectives_best, current_min])

    return overall_best, current_min - overall_best


def get_best_maximize(
    objective_values: Vector, current_max: Real
) -> Tuple[Real, float]:
    """Get the best from a list of objectives and the current best value.

    Args:
        objective_values: Vector of objectives.
        current_max: current maximum objective value.

    Returns:
        (Real, float), maximum found value and its difference with the current maximum.
    """
    objectives_max = np.max(objective_values)

    overall_best = np.max([objectives_max, current_max])

    return overall_best, overall_best - current_max


def top_n_indices(x: Vector, sense: str, top_n: int) -> np.ndarray:
    """
    Utility for find the indices of the `top_n`
    elements of a vector, when that vector is
    sorted according to `sense`.

    Argsort sorts in ascending order. For maximize,
    we take the last `top_n` elements.
    For minimization, we sort then take the first
    `top_n` elements.

    Args:
        x: Vector of values to sort and find best idx for.
        sense: Whether to maximize or minimize the function
        top_n: How many top solutions to check for constraint violations

    Returns:
        int

    """
    if sense == "maximize":
        best_idx = np.argsort(x)[-top_n:]
    elif sense == "minimize":
        best_idx = np.argsort(x)[:top_n]
    else:
        raise ValueError(f"Invalid sense {sense} provided.")
    return best_idx
