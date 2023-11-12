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
This submodule stores information about all model protocols that are used in reporting.
We are using structural subtyping to show interfaces of used models.
"""

import typing as tp

import numpy as np
import pandas as pd

from ._model_related import Estimator, ShapExplanation

TVector = tp.Union[pd.Series, np.ndarray]
_TFeatureImportanceDict = tp.Dict[str, float]


class SupportsBaseModel(tp.Protocol):
    """
    Implementation of ``modeling.ModelBase`` satisfies this protocol.

    This is a Protocol of base model that implements following methods:
        * ``predict``
        * ``get_feature_importance``
        * ``target``
        * ``features_in``
        * ``features_out``
    """

    def predict(self, data: pd.DataFrame, **kwargs: tp.Any) -> TVector:
        """
        Method for model prediction.

        Args:
            data: DataFrame to make a prediction
            **kwargs: Additional keyword arguments that
             might be passed for model prediction

        Returns:
            A Series or ndarray of model prediction
        """

    def get_feature_importance(
        self,
        data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> tp.Dict[str, float]:
        """
        An abstract method for getting feature importance from model.

        Args:
            data: DataFrame to build feature importance
            **kwargs: Additional keyword arguments
             that might be passed for model training

        Returns:
            Dict with feature names as a keys and feature importances as a values
        """

    @property
    def target(self) -> str:
        """
        Property containing model target column name.

        Returns:
            Column name
        """

    @property
    def features_in(self) -> tp.List[str]:
        """
        An abstract property containing columns that are required
        to be in the input dataset for ``ModelBase`` ``.fit`` or ``.predict`` methods.

        Returns:
            List of column names
        """

    @property
    def features_out(self) -> tp.List[str]:
        """
        A property containing names of the features, produced as the
        result of transformations of the inputs dataset
         inside ``.fit`` or ``.predict`` methods.

        If no dataset transformations are applied, then `.features_out` property
        returns same set of features as ``.features_in``.

        Returns:
            List of feature names
        """


class SupportsEvaluateMetrics(tp.Protocol):
    """
    Implementation of ``modeling.EvaluatesMetrics``
    satisfies this protocol.

    This is a Protocol of model that implements following methods:
        * ``evaluate_metrics``

    Notes:
    """

    def evaluate_metrics(
        self,
        data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> tp.Mapping[str, float]:
        """
        Define API for objects, that produce evaluation
        metrics based on the provided data

        Args:
            data: data to calculate metrics
            **kwargs: additional keyword arguments that
             are required for method implementation

        Returns:
            Mapping from metric name into metric value
        """


class SupportsShapFeatureImportance(tp.Protocol):
    """
    Implementation of ``modeling.ProducesShapFeatureImportance``
    satisfies this protocol.

    This is a Protocol of model that implements following methods:
        * ``produce_shap_explanation``
    """

    def produce_shap_explanation(
        self,
        data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> ShapExplanation:
        """
        Produce an instance of shap.Explanation based on provided data.

        Args:
            data: data to calculate SHAP values
            **kwargs: additional keyword arguments that
             are required for method implementation

        Notes:
            Columns in the resulting matrix should match
             original order of columns in provided data

        Returns:
            ``shap.Explanation`` containing prediction base values and SHAP values
        """

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

        Notes:
            Columns in the resulting dict should match
             original order of columns in provided data.

        Returns:
            Mapping from feature name into numeric feature importance
        """

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


class SupportsBaseModelAndEvaluateMetrics(
    SupportsBaseModel,
    SupportsEvaluateMetrics,
    tp.Protocol,
):
    """
    This is a Protocol that unifies following protocols:
        * ``SupportsBaseModel``
        * ``SupportsEvaluateMetrics``.

    See parental protocols for details.
    """


class SupportsModelFactoryBase(tp.Protocol):
    """
    Implementation of ``modeling.ModelFactoryBase`` satisfies this
     protocol.

    This is a Protocol of base model factory that implements following methods:
        * ``__repr__``

    Notes:
        This is currently used only for typing.
        This class is kept a placeholder so the needed methods from ModelFactoryBase can
        be added here when/if they are actually needed.
    """

    def __repr__(self) -> str:
        """
        String representation for model factory.

        Returns:
            String representation.
        """


class SupportsModelTunerBase(tp.Protocol):
    """
    Implementation of ``modeling.ModelTunerBase`` satisfies this
     protocol.

    This is a Protocol of base model tuner that implements following methods:
        * ``__repr__``

    Notes:
        This is currently used only for typing.
        This class is kept a placeholder so the needed methods from ``ModelTunerBase``
        can be added here when/if they are actually needed.
    """

    def __repr__(self) -> str:
        """
        String representation for model tuners.

        Returns:
            String representation.
        """


class Model(
    SupportsBaseModel,
    SupportsEvaluateMetrics,
    SupportsShapFeatureImportance,
    tp.Protocol,
):
    """
    This is a Protocol that unifies following protocols:
        * ``SupportsBaseModel``
        * ``SupportsEvaluateMetrics`.`
        * ``SupportsShapFeatureImportance`.`

    See parental protocols for details.
    """


class SklearnModel(SupportsBaseModel, tp.Protocol):
    """
    Implementation of `modeling.SklearnModel` satisfies this protocol.

    This is a Protocol of model that implements following methods:
        * ``estimator``

    And unifies following protocols:
        * ``SupportsBaseModel``

    See parental protocols for details.
    """

    @property
    def estimator(self) -> Estimator:
        """Returns sklearn estimator"""
