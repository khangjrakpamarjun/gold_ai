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
"""Variance inflation factor (vif) feature selection."""
# pylint: disable=attribute-defined-outside-init

from typing import Any, Optional

import numpy as np
import pandas as pd
import statsmodels as sm
from sklearn.utils.validation import check_is_fitted
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ..base import Transformer


class VifSelector(Transformer):
    """A selector performing recursive, variance inflation factor feature selection.
    Attributes:
        threshold: The maximum variance inflation factor allowed.
        support_: A numpy array/boolean mask of which columns are selected
        selected_features_: list of features that are selected.

    """

    def __init__(self, threshold: float = 10):
        """Instantiate the selector and attach parameters.

        Args:
            threshold: The maximum variance inflation factor allowed
        """
        self.threshold = threshold

    def fit(
        self,
        x: pd.DataFrame,  # noqa: WPS111
        y: Optional[pd.DataFrame] = None,  # noqa: WPS111
        **fit_params: Any,
    ) -> "VifSelector":
        """Fits the selector.
        The features are selected by recursively removing the feature with the highest
        variance inflation factor until the highest factor is below the threshold or
        only one feature is left.

        Args:
            x: The data for which we will select features
            y: Unused, for compatibility only
            **fit_params: Unused, for compatibility only

        Returns:
            A fitted selector
        """
        self.check_x(x)
        data = x.copy()
        all_columns = list(data.columns)
        selected_columns = list(data.columns)

        while True:
            factor_df = estimate_variance_inflation_factors(data)
            factor_df = factor_df.sort_values(
                "variance_inflation_factor",
                ascending=False,
            )
            if factor_df.variance_inflation_factor.values[0] >= self.threshold:
                # Pick the feature with the highest VIF
                col = factor_df.feature_name.values[0]
                selected_columns.remove(col)
                data = data.drop(col, axis=1)

                if len(selected_columns) <= 1:
                    break

            else:
                break
        self.support_ = np.array([col in selected_columns for col in all_columns])
        self.selected_features_ = selected_columns
        return self

    @property
    def threshold(self):
        """Get vif threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        """Validate and set vif threshold."""
        if threshold <= 1.0:
            raise ValueError("Threshold should be larger than 1")
        self._threshold = threshold

    def get_support(self) -> np.ndarray:
        """Provides a boolean mask of the selected features."""
        check_is_fitted(self)
        return self.support_

    def transform(self, data: pd.DataFrame):
        """Selects the features selected in `fit` from a provided dataframe."""
        check_is_fitted(self)
        self.check_x(data)
        return data.loc[:, self.support_]


def estimate_variance_inflation_factors(  # pylint:disable=invalid-name
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create a table of variance inflation factors, sorted descending.
    Categorical variable case not explored yet.

    Args:
        df: pandas dataframe with selected features.

    Returns:
        Pandas dataframe containing all columns and associated VIF.

            +------------+-------------------------+
            |feature_name|variance_inflation_factor|
            +------------+-------------------------+
            |    feature1|                      8.1|
            |    feature2|                      5.3|
            |    feature3|                      2.6|
            |    feature4|                      1.2|
            +------------+-------------------------+
    """
    # Intercept needed for statsmodels
    df = sm.tools.tools.add_constant(df)

    vif = [
        variance_inflation_factor(
            np.array(df),
            df.columns.get_loc(column),
        )
        for column in df.columns
    ]
    vif_pd = pd.DataFrame(
        {"feature_name": df.columns, "variance_inflation_factor": vif},
    )

    # Remove value for intercept
    vif_pd = vif_pd.loc[vif_pd.feature_name != "const"]

    return vif_pd.sort_values(
        "variance_inflation_factor",
        ascending=False,
    ).reset_index(drop=True)
