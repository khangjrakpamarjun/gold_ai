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


# MEMO: Implement an instance of class {class_name}, so you do not have to deal with "a
#  vs an" issues and you can implement the method in the base class
# MEMO: Benchmark models should not belong within reporting chart, but in their own
#  place in reporting
#  Then the charts should import them and use them where needed
from __future__ import annotations

import typing as tp
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ._regression_metrics import evaluate_regression_metrics

TVector = tp.Union[pd.Series, np.ndarray]
TMatrix = tp.Union[np.ndarray, np.matrix, pd.DataFrame]
TBenchmarkModel = tp.TypeVar("TBenchmarkModel", bound="BenchmarkModelBase")


def check_columns_exist(
    data: pd.DataFrame,
    col: tp.Union[str, tp.Iterable[str]] = None,
) -> None:
    """Raises an error if columns with the name(s) provided in ``col`` are not present
    in ``data``.
    """
    if isinstance(col, str):
        if col not in data.columns:
            raise ValueError(f"Column {col} is not included in the dataframe.")

    if isinstance(col, tp.Iterable):
        columns = set(col)
        if not columns.issubset(data.columns):
            not_included_cols = columns - set(data.columns)
            raise ValueError(
                "The following columns are missing"
                f" from the dataframe: {not_included_cols}.",
            )


def _check_input_len_equals_prediction_len(data: TMatrix, prediction: TVector) -> None:
    if len(data) != len(prediction):
        raise ValueError("Length of input data is not the same as prediction length.")


# TODO: Understand and decide what to do with timestamp, target and features_in
#  For now the assumption is that the benchmark models will only require `timestamp`
#  and `target`.
#  For now `_features_in` is kept as internal variable
class BenchmarkModelBase(ABC):
    """Abstract class for benchmark models.

    Benchmark models are unsophisticated models intended to provide a benchmark for
    model performance when modeling.
    """

    def __init__(
        self,
        target: str,
        timestamp: str,
    ) -> None:
        self._target = target
        self._timestamp = timestamp
        self._features_in = [self._timestamp, self._target]

    def predict(self, data: pd.DataFrame, **kwargs: tp.Any) -> TVector:
        """
        A public method for model prediction.

        Args:
            data: DataFrame to make a prediction
            **kwargs: Additional keyword arguments that might be passed for model
             prediction

        Returns:
            A Series or ndarray of model prediction
        """
        check_columns_exist(data, col=self._features_in)
        class_name = self.__class__.__name__
        _warn_if_nans_are_present(
            data[self.features_in],
            message=(
                "Nulls detected in relevant data when predicting using an instance of"
                f"`{class_name}`. The prediction might be inaccurate."
            ),
        )
        prediction = self._predict(data, **kwargs)
        _check_input_len_equals_prediction_len(data, prediction)
        return prediction

    def fit(
        self: TBenchmarkModel,
        data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> TBenchmarkModel:
        """
        A public method to train model with data.

        Args:
            data: DataFrame to train model on
            **kwargs: Additional keyword arguments
             that might be passed for model training

        Returns:
            A trained instance of BaseModel class
        """
        check_columns_exist(data, col=self._features_in)
        class_name = self.__class__.__name__
        _warn_if_nans_are_present(
            data[self.features_in],
            message=(
                "Nulls detected in relevant data when fitting an instance of"
                f"`{class_name}`."
            ),
        )
        return self._fit(data, **kwargs)

    def evaluate_metrics(
        self,
        data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> tp.Dict[str, float]:
        """
        Calculate standard set of regression metrics:
            * Mean absolute error
            * Rooted mean squared error
            * Mean squared error
            * Mean absolute percentage error
            * R^2` (coefficient of determination)
            * Explained variance
        Args:
            data: data to calculate metrics
            **kwargs: keyword arguments passed to ``.predict`` method
        Returns:
            Mapping from metric name into metric value
        Notes:
            This method is intended to be compatible with the ``evaluate_metrics``
             method, defined according to the ``optimus_core.EvaluatesMetrics`` abstract
             base class
        """
        target = data[self.target]
        prediction = self.predict(data)
        return evaluate_regression_metrics(target, prediction)

    # TODO: Fix this docstring in ModelBase, then update it here too
    @property
    def features_in(self) -> tp.List[str]:
        """
        A property containing columns that are required to be in the input dataset
         for ``BenchmarkModelBase`` ``.fit`` or ``.predict`` methods.

        Returns:
            List of column names
        """
        return self._features_in.copy()

    @property
    @abstractmethod
    def features_out(self) -> tp.List[str]:
        """
        An abstract property containing names of the features, produced as the result of
         transformations of the inputs dataset inside ``.fit`` or ``.predict`` methods.

        If no dataset transformations are applied, then ``.features_out`` property
         returns same set of features as ``.features_in``.

        Returns:
            List of feature names
        """

    @property
    def target(self) -> str:
        """
        An abstract property containing model target column name.

        Returns:
            Column name
        """
        return self._target

    @property
    def timestamp(self):
        """Returns the name of the timestamp column.

        Allows the user to know which of the feature has the "special" role of being
        the timestamp.
        """
        return self._timestamp

    @abstractmethod
    def __repr__(self) -> str:
        """
        An abstract method for string representation for models.

        Notes:
            As per definition of ``__repr__`` we strive to return a string
            representation that would yield an object with the same value when passed to
            eval();

        Returns:
            String representation.
        """

    @abstractmethod
    def _predict(self, data: pd.DataFrame, **kwargs: tp.Any) -> TVector:
        """
        An abstract method for model prediction logic implementation.

        Args:
            data: DataFrame to make a prediction
            **kwargs: Additional keyword arguments that might be passed for model
             prediction

        Returns:
            A Series or ndarray of model prediction
        """

    @abstractmethod
    def _fit(
        self: TBenchmarkModel,
        data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> TBenchmarkModel:
        """
        An abstract method for model training implementation.

         Args:
            data: DataFrame to train model on
            **kwargs: Additional keyword arguments that might be passed for model
             training

        Returns:
            A trained instance of BaseModel class
        """


def _warn_if_nans_are_present(data: pd.DataFrame, message: str) -> None:
    """Raises a warning if NaNs are present.

    Notes:
        The original use intended for this function is to warn of NaNs in
        ``data[self._feature_in]`` in the benchmark model
        Uses the provided ``message`` as warning message
    """
    if data.isnull().any(axis=None):
        warnings.warn(message)
