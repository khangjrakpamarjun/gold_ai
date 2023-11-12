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
"""A simple wrapper for scikit-learn feature selectors to ease working with
pandas dataframes.
From brix: (see the quantumblack/brix repo on Github)
pandas_selector_wrapper/src/
pandas_selector_wrapper.py
"""
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

from ..base import Transformer


class SkLearnSelector(Transformer):
    """
    A wrapper for scikit-learn feature selectors that preserves dataframes as outputs,
    and stores the names of the input and output features.

    Attributes:
        input features_: list of input feature names.
        selected_features_: list of features that are selected.
        selector: the scikit-learn selector object that is used.
    """

    def __init__(self, selector: SelectorMixin):
        """
        Args:
            selector: a scikit-learn feature selector object.
        """
        self.selector = selector

    def fit(
        self,
        x: pd.DataFrame,  # noqa: WPS111
        y: Optional[Union[pd.Series, np.ndarray]] = None,  # noqa: WPS111
        **fit_params,
    ) -> "SkLearnSelector":
        """
        Fit this selector.

        After this method has been called, the ``input_features_`` and
        ``selected_features_`` attributes will be set.

        Args:
            x: input dataframe.
            y: the target variable.
            **fit_params: keyword arguments that are passed to ``self.selector.fit``

        Returns:
            this selector, now fit.

        """
        self._check_input(x)
        self.input_features_ = list(x.columns)  # noqa: WPS120
        self.selector.fit(x[self.input_features_], y=y, **fit_params)
        self.selected_features_ = [  # noqa: WPS120
            feature
            for feature_index, feature in enumerate(self.input_features_)
            if feature_index in self.selector.get_support(indices=True)
        ]

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:  # noqa: WPS111
        """
        Given an input dataframe containing all of the features, return a copy
        containing only the selected features.

        Args:
            x: input dataframe containing all of the features.

        Returns:
            dataframe containing only the selected features.

        """
        check_is_fitted(self, "selected_features_")
        self._check_input(x, is_train=False)
        return x[self.selected_features_].copy()

    def _check_input(self, x, is_train=True):  # noqa: WPS111
        self.check_x(x)

        # TODO: Refactor multiline condition
        if not is_train and not all(  # noqa: WPS337
            feature in x.columns for feature in self.input_features_
        ):
            raise ValueError(
                "All of ``input_features`` must be columns in the input dataframe.",
            )

        if not is_train and set(x.columns) - set(self.input_features_):
            extra_columns = sorted(set(x.columns) - set(self.input_features_))
            warnings.warn(
                f"Your input dataframe contains unexpected extra columns:"
                f" {extra_columns}.",
                UserWarning,
            )
