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


import typing as tp
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .utils import check_columns_exist

_TFeatureImportanceDict = tp.Dict[str, float]


class ShapExplanation(tp.Protocol):
    """
    Type stub protocol for `shap.Explanation`
    """

    @property
    def values(self) -> np.array:  # noqa: WPS110
        """
        `np.array` of SHAP values
        """

    @property
    def data(self) -> np.array:
        """
        `np.array` of original data called to be explained
        """

    @property
    def base_values(self) -> np.array:
        """
        `np.array` of SHAP base values – E(F(X)).
        """

    @property
    def feature_names(self) -> np.array:
        """
        `np.array` of column names from data
        """


class ProducesShapFeatureImportance(ABC):
    """Define API for objects, that produce SHAP values based on provided data."""

    def __init__(
        self,
        features_in: tp.Iterable[str],
        *args: tp.Any,
        **kwargs: tp.Any,
    ) -> None:
        self._features_in = list(features_in)

    def produce_shap_explanation(
        self,
        data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> ShapExplanation:
        """
        Check that ``self._features_in`` feature set
        exists in provided data and return the SHAP explanation.

        Args:
            data: data to calculate SHAP
             values containing ``self._features_in`` feature set
            **kwargs: additional keyword arguments that
             are required for method implementation

        Returns:
            `shap.Explanation` containing prediction base values and SHAP values
        """
        check_columns_exist(data, col=self._features_in)
        return self._produce_shap_explanation(data[self._features_in], **kwargs)

    def get_shap_feature_importance(
        self,
        data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> _TFeatureImportanceDict:
        """
        Calculate feature importance from SHAP values.

        Notes:
            This method consecutively calls two methods:
            first `produce_shap_explanation` and then
            `get_shap_feature_importance_from_explanation`

            So if you are producing shap explanations by
            `produce_shap_explanation` (lets say, to build a shap summary plot),
            you can reduce one extra shap explanation calculation, using
            `get_shap_feature_importance_from_explanation` with a pre-calculated
            explanation instead of calling this method.

        Args:
            data: data to calculate SHAP values
            **kwargs: keyword arguments to
             pass into ``produce_shap_explanation``

        Returns:
            Mapping from feature name into numeric feature importance
        """
        explanation = self.produce_shap_explanation(data, **kwargs)
        return self.get_shap_feature_importance_from_explanation(explanation)

    @staticmethod
    def get_shap_feature_importance_from_explanation(
        explanation: ShapExplanation,
    ) -> _TFeatureImportanceDict:
        """
        Calculate feature importance from the provided SHAP explanation.
        Importance is calculated as a mean absolute shap value.

        Args:
            explanation: shap-explanation object

        Returns:
            Mapping from feature name into numeric feature importance
        """
        shap_feature_importance = np.abs(explanation.values).mean(axis=0)
        return dict(zip(explanation.feature_names, shap_feature_importance))

    @abstractmethod
    def _produce_shap_explanation(
        self,
        data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> ShapExplanation:
        """
        Produce an instance of shap.Explanation
        based on provided data for ``self._features_in`` feature set.

        Args:
            data: data to calculate SHAP
             values containing ``self._features_in`` feature set
            **kwargs: additional keyword arguments that
             are required for method implementation

        Returns:
            `shap.Explanation` containing prediction base values and SHAP values
        """
