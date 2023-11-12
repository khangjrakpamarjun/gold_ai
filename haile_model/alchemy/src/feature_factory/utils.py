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
"""This module provides a set of helper functions being used in the package.
"""
import importlib
from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DIMENSIONALITY_ERROR_PANDAS = (
    "Expected DataFrame, got a Series instead.\n"
    "Convert your data to a DataFrame using pd.DataFrame(series) if "
    "your data has a single feature or series.to_frame().T "
    "if it contains a single sample."
)


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


class Transformer(TransformerMixin, BaseEstimator, ABC):
    """Transformer base class"""

    @staticmethod
    def check_x(x: Any):  # noqa: WPS111
        """
        Checks that the input is a pandas dataframe.
        Throws a helpful, sklearn-like exception if it is a series.

        Args:
            x: input to check

        Raises:
            TypeError: if input is not a dataframe
        """
        if isinstance(x, pd.Series):
            raise TypeError(DIMENSIONALITY_ERROR_PANDAS)
        if not isinstance(x, pd.DataFrame):
            type_of_input = type(x)
            raise TypeError(f"x should be a dataframe, got {type_of_input}")

    def fit(
        self,
        x: pd.DataFrame,  # noqa: WPS111
        y: Union[np.ndarray, pd.Series] = None,  # noqa: WPS111
        **fit_params: Any,
    ):  # pylint:disable=unused-argument
        """
        Train the transformer on trainig data.

        Args:
            x: training data
            y: training y (optional)
            **fit_params: keyword args/params for call to fit

        Returns:
            self
        """
        return self

    @abstractmethod
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:  # noqa: WPS111
        """
        Transform x according to what was learned on the test data.

        Args:
            x: dataframe
        """
        raise NotImplementedError()
