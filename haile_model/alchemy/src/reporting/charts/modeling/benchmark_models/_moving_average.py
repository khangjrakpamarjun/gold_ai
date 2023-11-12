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

from ._benchmark_model_base import BenchmarkModelBase


class MovingAverageBenchmarkModel(BenchmarkModelBase):
    def __init__(
        self,
        target: str,
        timestamp: str,
        time_window: tp.Optional[str] = "30D",
        target_rolling_average_name: tp.Optional[str] = None,
    ) -> None:
        """
        ``target`` is the name of the column with the target values
        ``time_window`` specifies the time window over which the average is taken
        ``timestamp`` is the column with the timestamps
        ``target_rolling_average_name`` allows specifying a custom name for the rolling
            average feature

        It also defines ``_fitted_target_rolling_average``, initialized as ``None``,
        which will be used to store the moving average calculated from the data in the
        ``.fit`` step.
        """
        super().__init__(timestamp=timestamp, target=target)
        self._time_window = pd.tseries.frequencies.to_offset(time_window)
        self._target_rolling_average_name = (
            target_rolling_average_name
            if target_rolling_average_name is not None
            else f"{target}_rolling_average"
        )
        self._fitted_target_rolling_average = None

    @property
    def time_window(self) -> str:
        """Returns the string describing the time window used for the moving average"""
        return self._time_window.freqstr

    @property
    def features_in(self) -> tp.List[str]:
        """Returns the initial target column used for creating the moving average and
        the timestamp"""
        return self._features_in.copy()

    @property
    def features_out(self) -> tp.List[str]:
        """
        The ``features_out`` property for the ``MovingAverageBenchmarkModel`` class
         returns the name of the moving average feature
        """
        return [self._target_rolling_average_name]

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (
            f"{class_name}(target={self._target}, "
            f"timestamp={self._timestamp}, "
            f"time_window={self.time_window}, "
            f"target_rolling_average_name={self._target_rolling_average_name})"
        )

    def _make_common_preparation_steps(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Performs the preprocessing steps common to both fitting and predicting.

        These steps are:
        - ensure/enforce that the timestamp values are of ``datetime`` type
        - create a dataframe with only the ``self.timestamp`` and ``self._target``
            columns
        """
        timestamp_as_datetime = pd.to_datetime(data[self.timestamp])
        return data[[self.target, self.timestamp]].assign(
            **{
                self.timestamp: timestamp_as_datetime,
            }
        )

    def _fit(self, data: pd.DataFrame, **kwargs: tp.Any) -> MovingAverageBenchmarkModel:
        """Fits a moving-average benchmark model.

        The fitting step
        - uses the target and the timestamp columns
        - uses only rows where both the target and the timestamp are valid
        - sorts the data by increasing timestamp
        - calculates the moving average of the target
        - stores the moving averages (and their timestamps) into
            ``self._fitted_target_rolling_average`` for later use in prediction

        The moving averages are stored with the timestamp as index, so it is ready for
        the predict step later, where the time_window uses the dataframe index to
        determine which values to include in the rolling average.
        """
        df = (
            self._make_common_preparation_steps(data)
            .dropna()
            .set_index(self.timestamp)
            .sort_index()
        )
        self._fitted_target_rolling_average = (
            df[self.target]
            .rolling(
                self._time_window,
            )
            .mean()
        )
        return self

    def _predict(self, data: pd.DataFrame, **kwargs: tp.Any) -> pd.Series:
        """
        The prediction is one of the available moving averages calculated during fit.
        1. for timestamp t, the most recent moving average available at or before time t
            is used
        2. if timestamp t is before all the timestamps used in the train set,
            then the first (earliest) of the available averages is used
        The reindex with `method="ffill"` takes care of point 1., and the
          subsequent `.bfill()` takes care of point 2..
        """
        original_index = data.index
        df = self._make_common_preparation_steps(data).set_index(self.timestamp)
        y_pred = (
            self._fitted_target_rolling_average.reindex_like(
                df[self.target],
                method="ffill",
            )
            .bfill()
            .rename(self._target_rolling_average_name)
        )
        # Taking care of the edge case when the data are entirely before the train_data
        if max(df.index) < min(self._fitted_target_rolling_average.index):
            y_pred = self._fitted_target_rolling_average.reindex_like(
                df[self.target],
                method="bfill",
            )
        y_pred.index = original_index
        return y_pred
