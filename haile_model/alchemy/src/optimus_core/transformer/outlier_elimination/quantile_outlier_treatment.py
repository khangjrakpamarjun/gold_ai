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
from __future__ import annotations

import typing as tp

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin

TDict = tp.Dict[str, tp.Any]

BOTTOM_QUANTILE = 0.25
TOP_QUANTILE = 0.75
IQR_MULTIPLIER = 1.5


class QuantileOutlierRemoval(BaseEstimator, TransformerMixin):

    """
    Removes the outliers from the input dataframe given the
     method of removal and the limits accordingly.

    Attributes:
        how: method of removing outlier (currently
         supported methods:"IQR","quantiles",None)
        low_lim : lower cut off to be passed when "quantile"
         method is passed(between 0 - 1 range)
        up_lim : upper cut off to be passed when "quantile"
         method is passed(between 0 - 1 range)
        td: Tag Dictionary to be passed when neither "IQR" nor
         "quantile" methods are chosen.
    """

    def __init__(self, how: str, low_lim=None, up_lim=None, td: TDict = None):
        """
        Args:
             how: pass the allowed methods - currently supported
        'quantiles' : specify upper and lower limit for quantiles between 0 - 1 range
        'IQR': automatically assumes .75 and .25 as the upper and lower limit
        else if hard limit needs to be put,pass the tag dictionary from which range_min
         and range_max of the columns would be fetched.
        """

        self.how = how
        self.low_lim = low_lim
        self.up_lim = up_lim
        self.td = td
        self.lower_bounds = []
        self.upper_bounds = []

    def update_limits(self, data: pd.Series):  # noqa: WPS231
        """
        This function updates the lower_bounds and upper_bounds lists
         with lower and upper limit of the input column

        Args:
            data: input series

        """
        if self.how == "quantiles" and (self.low_lim is None or self.up_lim is None):
            raise ValueError(
                "With Quantiles as the method, give the low_lim and"
                " up_lim values while instantiating the QuantileOutlierRemoval",
            )

        if self.how is None and self.td is None:
            raise ValueError(
                "With 'how',the method of removal given as None, pass the"
                " tag Dictionary while instantiating the QuantileOutlierRemoval ",
            )
        if is_numeric_dtype(data):
            if self.how == "IQR":
                q1 = data.quantile(BOTTOM_QUANTILE)
                q3 = data.quantile(TOP_QUANTILE)
                iqr = q3 - q1
                lower_cutoff = q1 - (IQR_MULTIPLIER * iqr)
                upper_cutoff = q3 + (IQR_MULTIPLIER * iqr)
            elif self.how == "quantiles":
                lower_cutoff, upper_cutoff = (
                    data.quantile(self.low_lim),
                    data.quantile(self.up_lim),
                )
            else:
                lower_cutoff, upper_cutoff = (
                    self.td[data.name]["range_min"],
                    self.td[data.name]["range_max"],
                )
            self.lower_bounds.append(lower_cutoff)
            self.upper_bounds.append(upper_cutoff)
        else:
            self.lower_bounds.append(None)
            self.upper_bounds.append(None)

    def fit(
        self,
        x: pd.DataFrame,  # noqa: WPS111
        y=None,  # noqa: WPS111
    ) -> QuantileOutlierRemoval:
        """

        Args:
            x: input dataframe.

        """

        x.apply(self.update_limits, result_type="expand")
        return self

    def transform(
        self,
        x: pd.DataFrame,  # noqa: WPS111
        y=None,  # noqa: WPS111
    ) -> pd.DataFrame:
        """
        Given an input dataframe, removes the extreme outliers
         using the upper and lower limits

        Args:
            x: input dataframe.

        Returns:
            dataframe with outliers set to np.nan

        """

        for column_index in range(x.shape[1]):
            col = x.iloc[:, column_index].copy()
            if is_numeric_dtype(x[x.columns[column_index]]):
                col[
                    (col < self.lower_bounds[column_index])  # noqa: WPS465
                    | (col > self.upper_bounds[column_index])
                ] = np.nan
                x.iloc[:, column_index] = col
        return x
