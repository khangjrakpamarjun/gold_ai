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

import pandas as pd
from sklearn.linear_model import LinearRegression

from ._benchmark_model_base import BenchmarkModelBase


class AutoregressiveBenchmarkModel(BenchmarkModelBase):
    """
    A benchmark model that predicts the values of the target variable using a linear
     regression on the value it had ``order`` steps before.
    """

    def __init__(self, target: str, timestamp: str, order: int = 1) -> None:
        if order <= 0:
            raise ValueError(f"Model order must be >= 1, passed {order}")
        super().__init__(timestamp=timestamp, target=target)
        self.order = order
        self._linear_regression = LinearRegression()
        self._target_lag_name = f"{self._target}_lag_{self.order}"

    @property
    def features_in(self) -> tp.List[str]:
        """Returns initial target column used for creating lagged column feature"""
        return self._features_in.copy()

    @property
    def features_out(self) -> tp.List[str]:
        """
        Returns lagged column name used in model.
        Since we use a lagged target as a feature, we explicitly show that this is not
         the same target column we receive as an input.
        """
        return [self._target_lag_name]

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (
            f"{class_name}(target={self._target}, "
            f"timestamp={self._timestamp}, "
            f"order={self.order})"
        )

    def _keep_needed_columns_only(self, data: pd.DataFrame) -> pd.DataFrame:
        """Creates a table with two columns: timestamp and target.

        This preparation is common between ``_fit`` and ``_predict``
        """
        if self._target not in data.columns:
            raise ValueError(
                f"This is an autoregressive benchmark model. "
                f"Please provide target column ({self._target}) in the ``data``",
            )
        return pd.DataFrame(
            {self._timestamp: data[self._timestamp], self._target: data[self._target]},
        )

    # TODO: Consider two regression, one for forward prediction and one for backward
    #     prediction, and use the latter for the initial points
    def _predict(self, data: pd.DataFrame, **kwargs: tp.Any) -> pd.Series:
        """
        Predict using the fitted autoregressive model.

        Notes:
          When predicting,
          - prepare a dataframe with timestamp and target
          - drop index to be able to use an increasing int index to reorder data at the
              end
          - sort the data by timestamp
          - create the lagged targed by shifting the data by ``self.order``
          - forward fill the lagged data (assuming the target remains the same until
              changed)
          - backfill the lagged data (this allows predictions for the initial values,
              will introduce some bias though, because some point will be predicted by
              itself. Assuming this has a negligible effect for the purpose of a
              benchmark model)
          - restores the original order of the data
          - restore the original index (at this point the data have the same order and
              the same index as the input, but an additional column with the correct
              value of the lagged target is available)
          - predict the target using the linear model learned during fit
        """
        original_index = data.index.copy()
        df = (
            self._keep_needed_columns_only(data)
            .reset_index(drop=True)
            .sort_values(by=self._timestamp)
        )
        df = (
            df.assign(**{self._target_lag_name: df[self._target].shift(self.order)})
            .ffill()
            .bfill()
        )
        # So the prediction has the same order as the input, even with unordered input
        df = df.sort_index().set_index(original_index)
        prediction = self._linear_regression.predict(df[self.features_out])
        return pd.Series(prediction, index=original_index, name=self._target)

    def _fit(
        self,
        data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> AutoregressiveBenchmarkModel:
        """
        Fit an autoregressive model, based on the value of the target ``self.order``
        places before

        Notes:
          This method:
          - drops nulls
          - makes sure data are sorted by increasing timestamp
          - creates a column with a the target lagged by ``self.order`` positions
          - fits a linear regression predicting the target by using its lagged values
        """
        df = (
            self._keep_needed_columns_only(data)
            .dropna()
            .set_index(self._timestamp)
            .sort_index()
        )
        df = df.assign(
            **{self._target_lag_name: df[self._target].shift(self.order)},
        ).dropna()
        self._linear_regression.fit(df[self.features_out], df[self.target])
        return self
