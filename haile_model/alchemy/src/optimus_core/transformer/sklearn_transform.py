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
Sklearn Transformers wrapper for Pandas compatibility
"""
from __future__ import annotations

import typing as tp

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

from .base import Transformer


class ColumnNamesAsNumbers(Transformer):
    """
    Wrapper for sklearn-compatible transformers
    with a ``.fit`` and ``.transform`` methods, that converts
    the output into ``pd.DataFrame`` and sets column names
    equal to column position in the output matrix.

    Notes:
        This transformer does not assume
        that set of features before and after
        the transformation will be the same.

        If transformer preserves the features set and order,
        consider using the ``SklearnTransform`` instead.
    """

    def __init__(self, transformer: tp.Union[TransformerMixin, BaseEstimator]) -> None:
        self.transformer = transformer

    def fit(
        self,
        x: pd.DataFrame,  # noqa: WPS111
        y: tp.Union[np.ndarray, pd.Series] = None,  # noqa: WPS111
        **fit_params: tp.Any,
    ) -> ColumnNamesAsNumbers:
        """
        Checks input datasets and fits wrapped sklearn-compatible transformer.

        Args:
            x: Input data to train sklearn-compatible transformer
            y: Optional input target to train sklearn-compatible transformer

        Returns:
            Modified instance with trained wrapped sklearn-compatible transformer

        """
        self.check_x(x)
        self.transformer.fit(x, y, **fit_params)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:  # noqa: WPS111
        """
        Transforms using the wrapped transformer
        and wraps the output into the ``pd.DataFrame``
        """
        self.check_x(x)
        transformed_data = self.transformer.transform(x)
        _, transformed_data_columns_count = transformed_data.shape
        return pd.DataFrame(
            transformed_data,
            columns=range(transformed_data_columns_count),
            index=x.index,
        )


class SklearnTransform(Transformer):
    """
    Wrapper for sklearn-compatible transformers
    with a ``.fit`` and ``.transform`` methods, that converts
    the output into ``pd.DataFrame`` with the set of features
    taken from input ``pd.DataFrame``

    Notes:
        This transformer assumes that set of columns
         before and after the transformation will be the same.
    """

    def __init__(self, transformer: tp.Union[TransformerMixin, BaseEstimator]) -> None:
        self.transformer = transformer

    def fit(
        self,
        x: pd.DataFrame,  # noqa: WPS111
        y: tp.Optional[tp.Union[pd.Series, pd.DataFrame]] = None,  # noqa: WPS111
        **fit_params,
    ) -> SklearnTransform:
        """
        Checks input datasets and fits wrapped sklearn-compatible transformer.

        Args:
            x: Input data to train sklearn-compatible transformer
            y: Optional input target to train sklearn-compatible transformer

        Returns:
            Modified instance with trained wrapped sklearn-compatible transformer

        """
        self.check_x(x)
        self.transformer.fit(x, y, **fit_params)
        return self

    def transform(self, x: pd.DataFrame):  # noqa: WPS111
        """
        Transforms using the wrapped transformer
        and wraps the output into the ``pd.DataFrame``
        using the column names before transformation.
        """
        self.check_x(x)
        transformed_data = self.transformer.transform(x)
        _, input_data_columns_count = x.shape
        _, transformed_data_columns_count = transformed_data.shape
        if input_data_columns_count != transformed_data_columns_count:
            raise RuntimeError(
                "Columns count after the transformation"
                " does not match columns count of the input data:"
                f" input data contains {input_data_columns_count} columns,"
                f" and transformed data contains {transformed_data_columns_count}.",
            )
        return pd.DataFrame(transformed_data, columns=x.columns, index=x.index)


class OneHotEncoder(SklearnOneHotEncoder):
    """
    Customised Sklearn OneHotEncoder which accecpts the whole dataset and names of
    columns to be one hot encoded and returns the full dataframe with encoded columns

    Args:
        SklearnOneHotEncoder: Sklearn OneHotEncoder class
    """

    def __init__(self, columns: tp.List[str], **kwargs):
        """
        Constructor.

        Args:
            columns: a list of names of columns to be one hot encoded
        """
        if not columns:
            raise ValueError("Please privide names of columns to be one hot encoded!")
        super().__init__(**kwargs)
        self.columns = columns

    def fit(
        self,
        x: pd.DataFrame,  # noqa: WPS111
        y: tp.Union[pd.Series, pd.DataFrame] = None,  # noqa: WPS111
    ) -> None:
        """
        Calls the fit function of Sklearn One Hot Encoder

        Args:
            x: training data
            y: training y (no effect)

        Returns:
            self
        """
        self._check_columns(x)

        x_to_transform = x.loc[:, self.columns]
        super().fit(x_to_transform, y)

        return self

    def transform(self, x: pd.DataFrame):  # noqa: WPS111
        """
        Transforms specified columns of x with one hot encoder and concatenate
        one hot encoded columns with original dataframe x

        Args:
            x: training data

        Returns:
            pd.DataFrame
        """
        self._check_columns(x)

        x_to_transform = x.loc[:, self.columns]
        transformed = super().transform(x_to_transform)
        if not isinstance(transformed, np.ndarray):
            transformed = transformed.toarray()

        transformed_column_names = self.get_feature_names(input_features=self.columns)
        transformed = pd.DataFrame(transformed, columns=transformed_column_names)

        return pd.concat([x, transformed], axis=1).drop(self.columns, axis=1)

    def _check_columns(self, x: pd.DataFrame):  # noqa: WPS111
        """
        Check if columns exist in the input dataframe.

        Args:
            x: training data
        """
        if not all(col in x.columns for col in self.columns):
            missing_cols = set(self.columns) - set(x.columns)
            raise ValueError(f"Expected these columns in your dataframe {missing_cols}")
