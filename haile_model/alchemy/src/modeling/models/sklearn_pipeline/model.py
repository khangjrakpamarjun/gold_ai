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

import logging
import typing as tp
from copy import copy

import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from optimus_core.transformer import SelectColumns

from ... import api
from ..metrics_utils import evaluate_regression_metrics
from ..model_base import (
    EvaluatesMetrics,
    ModelBase,
    ProducesShapFeatureImportance,
    check_columns_exist,
)
from ..sklearn_model import attribute_check_is_fitted, validate_estimator

logger = logging.getLogger(__name__)


class SklearnPipeline(  # noqa: WPS214
    ModelBase,
    ProducesShapFeatureImportance,
    EvaluatesMetrics,
):
    """
    A wrapper for ``sklearn.pipeline.Pipeline`` with OAI related functionality.

    Note:
        Wrapped ``sklearn.pipeline.Pipeline`` might not contain
        feature selecting step explicitly.
        Features passed as ``features_in`` will be used for feature selection
        inside ``fit``, `predict` and ``transform`` methods.

        Please note, that ``get_pipeline()`` will return
        a sklearn.pipeline.Pipeline with ``SelectColumns`` as a first step.
        This is required for backwards compatibility - other project components use
        ``sklearn.pipeline.Pipeline`` rather that this wrapper.
    """

    def __init__(
        self,
        estimator: Pipeline,
        features_in: tp.Iterable[str],
        target: str,
        features_out: tp.Optional[tp.Iterable[str]] = None,
    ) -> None:
        """
        Args:
            estimator: A ``sklearn.pipeline.Pipeline`` to wrap
            target: Target column name
            features_in: Features that will be selected from the data
            to fit or predict with ``sklearn.pipeline.Pipeline``
            features_out: Input features for the last step of the Pipeline.
        """
        super().__init__(features_in=features_in, target=target)
        validate_estimator(estimator)
        self._pipeline = estimator
        self._features_out = None if features_out is None else list(features_out)

    @property
    @attribute_check_is_fitted("estimator")
    def features_out(self) -> tp.List[str]:
        """
        Pipeline might perform any data transformations.
        This property returns the list of columns that are
        used by the last step of wrapped ``sklearn.pipeline.Pipeline``
        as the input features.

        Returns:
            Copy of a list of columns that are
            used by the last step of the wrapped pipeline.
        """
        if self._features_out is None:
            logger.warning(
                "`features_out` property was not set during initialization"
                " and `features_out` assumed to be equal to `features_in`.\n"
                " Please run `.fit` to automatically fill `features_out`"
                " based on data or reinitialize object with `features_out` argument.",
            )
            return self.features_in
        return copy(self._features_out)

    @property
    def estimator(self) -> api.Estimator:
        """
        Returns:
            Copy of the last step of the wrapped ``sklearn.pipeline.Pipeline``
        """
        return copy(self._pipeline[-1])

    def get_pipeline(  # noqa: WPS615
        self,
        select_features_step_name: str = "select_features",
    ) -> Pipeline:
        """
        Export wrapped ``sklearn.pipeline.Pipeline`` with
        additional ``SelectColumns`` step.

        Returns:
            Copy of the wrapped ``sklearn.pipeline.Pipeline``
        """
        pipeline_steps = copy(self._pipeline).steps
        select_columns = SelectColumns(self._features_in)
        return Pipeline(
            [
                (
                    select_features_step_name,
                    select_columns.fit(pd.DataFrame(columns=self._features_in)),
                ),
                *pipeline_steps,
            ],
        )

    @attribute_check_is_fitted("estimator")
    def transform(self, data: pd.DataFrame, **kwargs: tp.Any) -> pd.DataFrame:
        """
        Transform data using the wrapped ``sklearn.pipeline.Pipeline``.
        Use all the pipeline steps except for the very last one.
        If there is only one step in pipeline, then return data as is.

        Notes:
            If ``self.features_out`` attribute was not specified,
            then it will be filled based on the factual data.

        Raises:
            RuntimeError, if factual columns after transformations
            does not match with specified ``self.features_out``

        Args:
            data: DataFrame used for transform
            **kwargs: Additional keyword arguments that might be passed
            to wrapped pipeline ``.transform`` method

        Returns:
            Transformed DataFrame
        """
        check_columns_exist(data, col=self._features_in)
        data = data[self._features_in]
        if len(self._pipeline) > 1:
            transformed_data = self._pipeline[:-1].transform(data, **kwargs)
        else:
            transformed_data = data
        self._update_features_out_with_factual_columns(transformed_data)
        check_transformed_columns(transformed_data, self.features_out)
        return transformed_data

    @attribute_check_is_fitted("estimator")
    def get_feature_importance(
        self,
        data: pd.DataFrame,
        **kwargs,
    ) -> tp.Dict[str, float]:
        """
        Generate a feature importances.
        If the last step of the wrapped ``sklearn.pipeline.Pipeline`` has
        a ``feature_importances_`` property, then it will be used
        for extracting importances.
        If not, ``sklearn.inspection.permutation_importances``
        will be used instead.

        Note:
            This method returns mapping from ``features_out`` (feature set
            that is consumed by final pipeline's estimator step)
            to feature importance. This is expected because
            most of the feature importance extraction technics
            utilize estimators and return importance
            for feature set used by estimator.

            If ``self.features_out`` attribute was not specified,
            then it will be filled based on the factual data.

        Raises:
            RuntimeError, if factual columns after transformations does
            not match with specified ``self.features_out``

        Args:
            data: DataFrame that might be used for feature importances extraction
            **kwargs: Additional keyword arguments that might be passed
            to wrapped model ``.predict`` method

        Returns:
            Dict with ``features_out`` feature list as a keys
            and feature importances as a values
        """
        check_columns_exist(data, col=self._features_in)
        estimator = self.estimator
        if getattr(estimator, "feature_importances_", None) is not None:
            feature_importance = estimator.feature_importances_
        else:
            type_of_estimator = type(estimator)
            logger.info(
                f"Estimator of type {type_of_estimator} does not"
                " have `feature_importances_` "
                "using sklearn.inspection.permutation_importances instead.",
            )
            transformed_dataset = self.transform(data, **kwargs)
            feature_importance = permutation_importance(
                estimator,
                transformed_dataset,
                data[self._target],
            )["importances_mean"]
        return dict(zip(self.features_out, feature_importance))

    @attribute_check_is_fitted("estimator")
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
            * R^2 (coefficient of determination)
            * Explained variance

        Args:
            data: data to calculate metrics on
            **kwargs: keyword arguments passed to ``.predict`` method

        Returns:
            Mapping from metric name into metric value
        """
        target = data[self._target].values
        prediction = self.predict(data, **kwargs)
        return evaluate_regression_metrics(target, prediction)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        pipeline_representation = repr(self._pipeline)
        return (
            f"{class_name}("
            f"estimator={pipeline_representation}, "
            f'target="{self._target}" ,'
            f"features_in={self._features_in}, "
            f"features_out={self._features_out}"
            ")"
        )

    @attribute_check_is_fitted("estimator")
    def _produce_shap_explanation(
        self,
        data: pd.DataFrame,
        algorithm: str = "auto",
        **kwargs: tp.Any,
    ) -> shap.Explanation:
        """
        Produce an instance of shap.Explanation based on
        provided data using generic `shap.Explainer`.

        Args:
            data: data to extract SHAP values from
            algorithm: algorithm to use; parameter to pass
             into `shap.Explainer` initialization
            **kwargs: keyword arguments to be provided
             for `shap.Explainer` initialization

        Returns:
            `shap.Explanation` for ``self.features_in`` feature set
            containing prediction base values and SHAP values
        """
        return self._explain_using_generic_explainer(
            data=data,
            algorithm=algorithm,
            **kwargs,
        )

    def _fit(self, data: pd.DataFrame, **kwargs: tp.Any) -> SklearnPipeline:
        """
        Train wrapped ``sklearn.sklearn.Pipeline`` on a provided data.

        Note:
            If ``self.features_out`` attribute was not specified,
            then it will be filled based on the factual data.

        Args:
            data: DataFrame, that contains ``self.features_in``
             feature set to make predictions
            **kwargs: Key-word arguments to be passed to ``sklearn`` method ``.fit``

        Returns:
            A trained instance of ``SklearnPipeline`` class
        """
        if len(self._pipeline) > 1:
            transformed_data = self._pipeline[:-1].fit_transform(
                data[self._features_in],
                data[self.target],
                **kwargs,
            )
        else:
            transformed_data = data[self._features_in]
        self._update_features_out_with_factual_columns(transformed_data)
        check_transformed_columns(transformed_data, self._features_out)
        self._pipeline[-1].fit(transformed_data, data[self.target], **kwargs)
        return self

    def _predict(self, data: pd.DataFrame, **kwargs: tp.Any) -> api.Vector:
        """
        Make a prediction using a wrapped ``sklearn.pipeline.Pipeline`` model.
        Passed DataFrame is required to have columns ``self.features_in``

        Note:
            If ``self.features_out`` attribute was not specified,
            then it will be filled based on the factual data.

        Raises:
            RuntimeError, if factual columns after transformations
            does not match with specified ``self.features_out``

        Args:
            data: DataFrame, that contains ``self.features_in``
             feature set to make predictions
            **kwargs: Key-word arguments to be passed to ``sklearn`` method ``.predict``

        Returns:
            A Series or ndarray with prediction.
        """
        transformed_data = self.transform(data[self._features_in], **kwargs)
        return self._pipeline[-1].predict(transformed_data, **kwargs)

    def _explain_using_generic_explainer(
        self,
        data: pd.DataFrame,
        algorithm: str,
        **kwargs: tp.Any,
    ) -> shap.Explanation:
        explainer = shap.Explainer(
            model=self._pipeline.predict,
            masker=data[self._features_in],
            algorithm=algorithm,
            **kwargs,
        )
        explainer_class_name = explainer.__class__
        class_name = self.__class__
        logger.info(
            f"`Using model-agnostic` {explainer_class_name}` to extract SHAP values..."
            f" `shap` can't apply model-specific algorithms for {class_name}."
            " Consider switching to `SklearnModel` if computation"
            " time or quality don't fit your needs.",
        )
        return explainer(data[self._features_in])

    def _update_features_out_with_factual_columns(self, data) -> None:
        _, columns_count = data.shape
        transformed_columns = (
            list(data.columns)
            if getattr(data, "columns", None) is not None
            else list(map(str, range(columns_count)))
        )
        if self._features_out is None:
            logger.info(
                "`features_out` attribute is not specified."
                " Setting `features_out` based on factual data.",
            )
            self._features_out = transformed_columns


def check_transformed_columns(
    transformed_data: api.Matrix,
    expected_columns: tp.List[str],
) -> None:
    """
    Check that transformed_data has expected list of columns.
    Raises an error if columns do not match.
    Warns if transformed data doesn't have `columns` attribute.

    Args:
        transformed_data: data transformed with sklearn.pipeline.
         Pipeline transform to check columns in.
        expected_columns: columns specified by user and expected
         to be present in `transformed_data`
    """
    transformed_columns = getattr(transformed_data, "columns", None)
    if transformed_columns is not None:
        if expected_columns != list(transformed_columns):
            raise RuntimeError(
                "Columns in the data after transformation"
                " does not match with expected list of columns.",
            )
    else:
        logger.warning(
            "Transformed data does not keep columns.\n"
            "Validation of columns after transform was not conducted. "
            "Learn how to keep columns after transformations "
            "with `optimus_core.transformer`.",
        )
