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
import logging
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.impute._base import _BaseImputer  # NOQA
from sklearn.utils.validation import check_is_fitted

from optimus_core.transformer import SklearnTransform
from optimus_core.utils import load_obj

logger = logging.getLogger(__name__)


# TODO: Revisit imputer when refactoring usecases
class ModelBasedImputer(_BaseImputer):
    def __init__(
        self,
        model: SklearnTransform,
        target_name: str,
        model_kwargs: Dict[str, Union[str, int, float]] = None,
        copy=True,
    ):
        """
        Used to predict ommited data in dataframes.

        Args:
            model: Here you should add name of your imputer class
            target_name: The name of your target function
            model_kwargs: Model arguments, for example random_state=0
            copy: Data copy
        """

        super().__init__(missing_values=np.NaN, add_indicator=False)
        self._model_type = model
        self._model_kwargs = model_kwargs if model_kwargs else {}
        self._model = None
        self.copy = copy
        self.features = None
        self.target_name = target_name

    def fit(self, X):  # noqa: WPS111,N803
        """This is "fit" method for imputer. It is to follow syntax from sklearn.

        Args:
            X: data

        Returns:
            self
        """
        x_validated = self._validate_input(X)
        self.features = x_validated.drop(self.target_name, axis=1).columns
        x_train = x_validated.dropna().reset_index(drop=True)
        y_train = x_train[self.target_name]
        self._model = load_obj(self._model_type)(**self._model_kwargs)
        self._model.fit(x_train[self.features], y_train)
        return self

    def transform(self, X):  # noqa: WPS111,N803
        """This is "transform" method for imputer. It is to follow syntax from sklearn.

        Args:
            X: data

        Returns:
            self
        """
        check_is_fitted(self._model)
        x_validated = self._validate_input(X)
        target_nan_mask = x_validated[self.target_name].isnull()
        x_test = x_validated[target_nan_mask].drop(self.target_name, axis=1)
        x_validated[self.target_name][target_nan_mask] = self._model.predict(x_test)
        return x_validated

    def _validate_input(self, X):  # noqa: WPS111,N803
        """Validates input

        Args:
            X: data

        Raises:
            ValueError: if data contains non_numeric columns
            ValueError: if data contains null values
            ValueError: if target variable is missing from data

        Returns:
            data
        """
        x_numeric = X.select_dtypes(include=np.number)
        x_non_numeric = X.drop(x_numeric, axis=1)
        if x_non_numeric.shape[1] != 0:
            raise ValueError("X must be numeric")
        if self.target_name in X.columns:
            if X.drop(self.target_name, axis=1).isnull().sum().sum():
                raise ValueError("X mustn't have NaNs")
            if len(X[self.target_name].dropna()) == len(X):
                logger.warning("No missing values to impute")
        else:
            raise ValueError("Target variable must be in X")
        if self.copy:
            return X.copy()
        return X


def fit_numeric_imputer(
    data: pd.DataFrame,
    cols_list: List[str],
    transformer: SklearnTransform,
    **transformer_kwargs,
) -> List[SklearnTransform]:
    """
    Used to create a model that can impute missing numerical data in a given dataset.

    Args:
        data: dataframe to fit with
        transformer: A Sklearn compatible transformer
        cols_list: list of numerical columns to inpute
        **transformer_kwargs: kwargs for transformer

    Returns:
        the imputer
    """

    impute_data = data.copy()

    impute_data = impute_data[cols_list]

    # create a new transformer
    transformer_class = load_obj(transformer)
    transformer = SklearnTransform(transformer_class(**transformer_kwargs))
    transformer.fit(impute_data)
    return transformer


def transform_numeric_imputer(
    data: pd.DataFrame,
    transformer: SklearnTransform,
    cols_list: List[str],
) -> List[Union[pd.DataFrame, SklearnTransform]]:
    """
    Used to impute missing numerical data in a given dataset.

    We assume that the transformer has
    already been fit, and thus, using the passed transformer, the data
    is transformed.

    Note:
        Only numerical columns are imputed

    Args:
        data: input data
        transformer: A Sklearn compatible transformer
        cols_list: list of numerical columns to inpute

    Returns:
        dataframe without missing numeric vals
    """
    impute_data = data.copy()

    if transformer is not None and getattr(transformer, "fit_transform", None) is None:
        raise TypeError("Passed Transformer does not have a fit_transform method")

    impute_data.loc[:, cols_list] = transformer.transform(impute_data[cols_list])
    return impute_data


def interpolate_cols(
    data: pd.DataFrame,
    cols_list: List[str] = None,
    method: str = "linear",
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Used to interpolate data on columns as specified in params

    For more information on the arguments you can place in params visit:
    https://pandas.pydata.org/pandas-docs/stable/
    reference/api/pandas.DataFrame.interpolate.html

    Args:
        data: input data
        cols_list: list of columns. if None, running on every column in input data
        **kwargs: kwargs used for pandas.Series.interpolate

    Returns:
        dataframe with interpolated data
    """

    interpolated_data = data.copy()

    if cols_list is None:
        cols_to_interpolate = interpolated_data.select_dtypes("number").columns.tolist()
        logger.info(f"Interpolating all numeric columns using '{method}' method.")

    else:
        cols_to_interpolate = list(np.intersect1d(interpolated_data.columns, cols_list))

    if not cols_to_interpolate:
        raise ValueError(
            "There are no tags defined in the interpolation parameters with kwargs "
            "which match any of the columns in the supplied dataframe",
        )

    for col in cols_to_interpolate:
        interpolated_data[col] = interpolated_data[col].interpolate(
            method=method,
            **kwargs,
        )
        if cols_list is not None:
            logger.info(f"Interpolating '{col}' column using '{method}' method.")

    return interpolated_data
