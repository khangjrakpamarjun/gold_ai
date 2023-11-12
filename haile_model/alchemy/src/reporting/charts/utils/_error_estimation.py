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

import numpy as np
import pandas as pd


def get_error_range(
    data: pd.DataFrame,
    variable_name: str,
    estimator: str,
    error_method: str = "ci",
    error_level: int = 95,
    random_state: int = 42,
) -> pd.Series:
    variable_values = data[variable_name]
    if error_method == "pi":
        err_min, err_max = _get_percentile_interval(variable_values, error_level)
    elif error_method == "ci":
        rand_gen = np.random.RandomState(random_state)
        boots = pd.DataFrame(
            rand_gen.choice(variable_values, size=(variable_values.size, 10000)),
        ).agg(estimator, axis=0)
        err_min, err_max = _get_percentile_interval(boots, error_level)
    else:
        raise ValueError(f"error_method {error_method} is not supported")
    estimate = variable_values.agg(estimator)
    return pd.Series(
        {
            variable_name: estimate,
            f"{variable_name}_min": err_min,
            f"{variable_name}_max": err_max,
        }
    )


def _get_percentile_interval(
    data: tp.Union[np.ndarray, pd.DataFrame],
    width: int,
) -> tp.Tuple[float, float]:
    """Return a percentile interval from data of a given width."""
    edge = (100 - width) / 2
    percentiles = edge, 100 - edge
    percentile_left, percentile_right = np.percentile(data, percentiles)
    return percentile_left, percentile_right  # noqa: WPS331
