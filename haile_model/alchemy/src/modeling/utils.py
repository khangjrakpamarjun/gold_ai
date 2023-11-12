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
Model utility functions.
"""
import importlib
import logging
import typing as tp

import pandas as pd
from pydantic import BaseModel

from . import api

logger = logging.getLogger(__name__)


class ObjectInitConfig(BaseModel):
    class_name: str
    kwargs: tp.Optional[tp.Dict[str, tp.Any]]


def load_obj(obj_path: str, default_obj_path: str = "") -> tp.Any:
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


# TODO: remove and refactor usages
def check_model_features(
    td: tp.Optional[api.SupportsTagDict] = None,
    td_features_column: tp.Optional[str] = None,
    model_features: tp.Optional[tp.List[str]] = None,
) -> tp.List[str]:
    """Check

    Raises:
        ValueError: if a tag dictionary, tag dictionary indicator column, and a list of
            model features are all provided.
        ValueError: if a tag dictionary is provided but a tag dictionary indicator
            column is not given.

    Args:
        td: optional tag dictionary to get feature names.
        td_features_column: optional string name of feature indicator column in the
            tag dictionary. This should be a column of True/False values indicating
            which tags are to be used as features.
        model_features: optional list of strings to use as features in the model.

    Returns:
        List of string feature names.
    """
    if td is None and td_features_column is None and model_features is None:
        raise ValueError(
            "Must specify a way of selecting model features. "
            "Provide a tag dict and indicator column"
            " name or list of feature names.",
        )

    if td is not None:
        if td_features_column is None:
            raise ValueError(
                "Must provide a boolean column in the tag dictionary "
                "defining which features belong to estimator.",
            )

        model_features = td.select(td_features_column)

    return model_features


# TODO: move to preprocessing package.
def drop_nan_rows(
    data: pd.DataFrame,
    target_column: str,
    td: tp.Optional[api.SupportsTagDict] = None,
    td_features_column: tp.Optional[str] = None,
    model_features: tp.List[str] = None,
) -> pd.DataFrame:
    """Drop any row that contains a nan in the desired feature + target set.
    Must provide a tag dictionary and td indicator column or a list of model features.

    Args:
        data: dataframe to drop nans from.
        target_column: string name of target column.
        td: optional tag dictionary.
        td_features_column: optional string name of feature indicator column in the
            tag dictionary. This should be a column of True/False values indicating
            which tags we should consider for the nan check.
        model_features: optional list of model feature names.

    Returns:
        DataFrame with nan rows dropped.
    """
    model_features = check_model_features(td, td_features_column, model_features)
    n_samples_before = len(data)
    data = data.dropna(subset=model_features + [target_column])
    n_samples_after = len(data)
    n_samples_dropped = n_samples_before - n_samples_after
    logger.info(
        f"Dropping {n_samples_dropped}"
        " rows with NaN values."
        f" Original sample size was {n_samples_before}"
        f" and is now {n_samples_after}.",
    )
    return data
