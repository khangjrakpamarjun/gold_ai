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

"""This module provides a set of helper functions being used across different components
of optimus package.
"""
import importlib
import os
import uuid
from distutils.util import strtobool
from functools import partial, update_wrapper
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence
from sklearn.pipeline import Pipeline as SklearnPipeline


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.

    Args:
        obj_path: Path to an object to be extracted, including the object name.
        default_obj_path: Default object path. In the case this is provided, `obj_path`
        must be a single name of the object being imported.

    Returns:
        Extracted object.

    Raises:
        AttributeError: When the object does not have the given named attribute.

    Examples:
        Importing an object::

            >>> load_obj("sklearn.linear_model.Ridge")

        Importing using `default_obj_path`::

            >>> load_obj("Ridge", default_obj_path="sklearn.linear_model")
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    loaded_object = getattr(module_obj, obj_name, None)
    if loaded_object is None:
        raise AttributeError(
            f"Object `{obj_name}` cannot be loaded from `{obj_path}`.",
        )
    return loaded_object


class NonBooleanEnvVariableError(Exception):
    """Error raised when provided value in env var is not boolean-like."""


def env_var_to_bool(env_var_name: str) -> bool:
    env_var_value = os.environ.get(env_var_name, "false")
    try:
        return strtobool(env_var_value.lower())
    except ValueError:
        raise NonBooleanEnvVariableError(
            f"The value {env_var_value} for "
            f"env var {env_var_name} is not boolean-like.",
        )


def partial_wrapper(func: Callable, *args, **kwargs) -> Callable:
    """Enables user to pass in arguments that are not datasets when function is called
    in a Kedro pipeline e.g. a string or int value.
    Args:
        func: Callable node function
     Returns:
        Callable
    """
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def generate_run_id(data: pd.DataFrame, col_name: str = "run_id") -> pd.DataFrame:
    """
    Extract the parameters saved in conf
    Adds a universally unique identifier for each run.
    Args:
        data: original DataFrame
        col_name: name for column for run_id
    Returns:
        DataFrame with run_id added
    Raises:
        AttributeError: When the param does not exist
    """
    columns = data.columns
    data[col_name] = [str(uuid.uuid4()) for _ in range(len(data.index))]
    return data[[col_name, *columns]]


def calculate_control_sensitivity(
    data: pd.DataFrame,
    model: SklearnPipeline,
    params: Dict,
) -> pd.DataFrame:
    """
    Returns Control Sensitivity df, which can be used
    for calculating Total Control Sensitivity

    Total Control Sensitivity is a metric for measuring
    the responsiveness of predictions to a given set of
    control variables. This metric is useful for fitting models
    that need to be optimized as monitoring this metric
    (along side traditional metrics  like r2_score, mape etc.)
    reduces the potential for selecting a model that results
    in low uplift during optimization.

    A higher (relative) Total Control Sensitivity indicates
    greater responsiveness of the predictions to a given set
    of control variables (and vice-versa for a lower value).

    Total Control Sensitivity is calculated as the sum of
    Control Sensitivity across all all control variables.

    Control Sensitivity measures the responsiveness of
    predictions to a given control variable. For a given control
    variable i,

    pdp_range_{i} = max(pdp_{i}) - min(pdp_{i})

    Control Sensitivity_{i} = pdp_range_{i} / mean(target variable)

    where pdp_{i} refers to the partial dependence series for i and
    mean(target variable) is a scaling factor.

    The magnitude of Control Sensitivity_{i} is a function of
    pdp_range_{i}. The larger the pdp_range , the higher the
    Control Sensitivity.

    Examples
    --------

    Calculating Total Control Sensitivity

    >>> sensitivity_df = calculate_control_sensitivity(df, model, params)
    >>> sensitivity_df
                 pdp_min	 pdp_max   pdp_range	control_sensitivity
    Control_1	0.701214	0.701214	0.000000	           0.000000
    Control_2	0.701260	0.704965	0.003706	           0.005276
    Control_3	0.701307	0.706821	0.005514	           0.007851

    >>> total_control_sensitivity = sensitivity_df['control_sensitivity'].sum()
    >>> print(total_control_sensitivity)
    0.013127

    Args:
        data: df containing:
            - features needed for generating model predictions
            - target variable
        model: Fitted model
        params: Model parameters coming from parameters.yml file.
            Required keys:
            - control_vars , list
            - target_var , str
    Returns:
        sensitivity: pd.DataFrame
    """

    features = model["select_columns"].items
    model = model["estimator"]

    # check if controls used for fitting the model
    unused_controls_by_model = set(params["control_vars"]) - set(features)
    if unused_controls_by_model:
        raise ValueError(
            f"Controls not found in model features:" f" {unused_controls_by_model}",
        )

    # calculate sensitivity for each control
    sensitivity = {}
    for control_variable in params["control_vars"]:
        pdp, axes = partial_dependence(model, data[features], [control_variable])
        sensitivity[control_variable] = {
            "pdp_min": np.min(pdp),
            "pdp_max": np.max(pdp),
        }
    sensitivity = pd.DataFrame(sensitivity).T
    sensitivity["pdp_range"] = sensitivity["pdp_max"] - sensitivity["pdp_min"]
    sensitivity["control_sensitivity"] = (
        sensitivity["pdp_range"] / data[params["target_var"]].mean()
    )
    return sensitivity.sort_values("pdp_range", ascending=True)
