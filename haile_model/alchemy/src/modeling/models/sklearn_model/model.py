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

import functools
import logging
import typing as tp
from copy import copy

import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.utils.validation import check_is_fitted

from ... import api
from ..metrics_utils import evaluate_regression_metrics
from ..model_base import (
    EvaluatesMetrics,
    ModelBase,
    ProducesShapFeatureImportance,
    check_columns_exist,
)
from ..shap_utils import explain_sklearn_estimator_robustly

logger = logging.getLogger(__name__)


def validate_estimator(estimator: api.Estimator):
    mandatory_methods = ("predict", "fit", "set_params", "get_params")
    for method in mandatory_methods:
        if getattr(estimator, method, None) is None:
            raise RuntimeError(
                f"Provided estimator does not have {method} method implemented.",
            )


def attribute_check_is_fitted(attribute_name: str) -> tp.Callable[..., tp.Any]:
    """
    A class method decorator to check whether on not an attribute
    with name `attribute_name` passes the sklearn check_is_fitted check.
    The check is performed before the actual call of the wrapped function.

    Args:
        attribute_name: Name of the attribute being checked

    Raises:
        NotFittedError, if provided attribute does not pass check.
    """

    def decorator(func: tp.Callable[..., tp.Any]) -> tp.Callable[..., tp.Any]:
        @functools.wraps(func)
        def wrapper(self: tp.Any, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            check_is_fitted(getattr(self, attribute_name))
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class SklearnModel(ModelBase, ProducesShapFeatureImportance, EvaluatesMetrics):

    """
    A wrapper for scikit-learn compatible estimators with OAI related functionality.
    """

    def __init__(
        self,
        estimator: api.Estimator,
        features_in: tp.Iterable[str],
        target: str,
    ):
        """
        Args:
            estimator: ``sklearn`` estimator to wrap
            target: Name of the column used in model training
            features_in: A list of feature names that are used
            to select features for model training or prediction.
        """
        super().__init__(features_in=features_in, target=target)
        validate_estimator(estimator)
        self._estimator = estimator

    @property
    def features_out(self) -> tp.List[str]:
        return self.features_in

    @property
    def estimator(self) -> api.Estimator:
        """
        Returns:
            A copy of wrapped scikit-learn compatible estimator
        """
        return copy(self._estimator)

    @attribute_check_is_fitted("_estimator")
    def get_feature_importance(
        self,
        data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> tp.Dict[str, float]:
        """
        Generate a feature importances.
        If the wrapped model has a ``feature_importances_``
        property, that will be returned.
        If not, ``sklearn.inspection.permutation_importances``
        will be used instead.

        Args:
            data: DataFrame that might be used for feature importances extraction
            **kwargs: Additional keyword arguments that might be passed
            to wrapped model `.predict` method

        Returns:
            Dict with feature names as a keys and feature importances as a values
        """
        check_columns_exist(data, col=self._features_in)
        estimator = self.estimator
        if getattr(estimator, "feature_importances_", None) is not None:
            feature_importance = estimator.feature_importances_
        else:
            estimator_type = type(estimator)
            logger.info(
                f"Estimator of type {estimator_type} does not"
                " have `feature_importances_` "
                "using `sklearn.inspection.permutation_importances` instead.",
            )
            feature_importance = permutation_importance(
                estimator,
                data[self._features_in],
                data[self._target],
            )["importances_mean"]
        return dict(zip(self._features_in, feature_importance))

    @attribute_check_is_fitted("_estimator")
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
        """
        Returns:
            String representation for ``SklearnModel``
        """
        class_name = self.__class__.__name__
        estimator_representation = repr(self._estimator)
        return (
            f"{class_name}("
            f"estimator={estimator_representation}, "
            f'target="{self._target}" ,'
            f"features_in={self._features_in}"
            ")"
        )

    @attribute_check_is_fitted("_estimator")
    def _produce_shap_explanation(
        self,
        data: pd.DataFrame,
        algorithm: tp.Optional[str] = None,
        robust: bool = True,
        **kwargs: tp.Any,
    ) -> shap.Explanation:
        """
        Produce an instance of shap.Explanation based on
         provided data using generic ``shap.Explainer``.

        Args:
            data: data containing ``self.features_in`` feature set
             to extract SHAP values from
            algorithm: algorithm to use; parameter to pass into ``shap.Explainer``
             initialization
            robust: if ``True``, uses robust shap explanation for sklearn models.
            **kwargs: keyword arguments to be provided for ``shap.Explainer``
             initialization

        Notes:
            A robust shap explanation, depending on the kind of estimator used, in
             general requires a combination of:
             - the correct algorithm
             - the correct object passed as "estimator to be explained" (between
              ``estimator`` and ``estimator.predict``)
            These choices can both be handled internally by
             ``explain_sklearn_estimator_robustly``.
            The user can anyway force a specific algorithm to be used even by a robust
             explanation by setting the ``algorithm`` parameter to the desired not-None
             value (at the user's own discretion though, since this will not guarantee
             that the explanation will succeed though).

        Returns:
            ``shap.Explanation`` containing prediction base values and SHAP values
             for ``self.features_in`` feature set.
        """
        if algorithm is None and not robust:
            algorithm = "auto"
            logger.info(
                f"Algorithm was set to '{algorithm}'. In case another algorithm is "
                "preferred, use the `algorithm` parameter to specify a SHAP extraction "
                "algorithm compatible with your model.",
            )
        if robust:
            return explain_sklearn_estimator_robustly(
                data=data,
                estimator=self.estimator,
                features=self._features_in,
                algorithm=algorithm,
                **kwargs,
            )
        return self._explain_using_generic_explainer(
            data=data,
            algorithm=algorithm,
            **kwargs,
        )

    @attribute_check_is_fitted("_estimator")
    def _predict(self, data: pd.DataFrame, **kwargs: tp.Any) -> api.Vector:
        """
        Make a prediction using wrapped ``sklearn`` model.

        Args:
            data: DataFrame, that contains ``self.features_in``
             feature set to make predictions
            **kwargs: Key-word arguments to be passed to `sklearn` method `.predict`

        Returns:
            A Series or ndarray with prediction.
        """
        return self._estimator.predict(
            data[self._features_in],
            **kwargs,
        )

    def _fit(self, data: pd.DataFrame, **kwargs: tp.Any) -> SklearnModel:
        """
        A method to train wrapped model on a provided data.

        Args:
            data: DataFrame, that contains ``self.features_in``
             feature set to train model
            **kwargs: Key-word arguments to be passed to ``sklearn`` method ``.fit``

        Returns:
            A trained instance of `SklearnModel`
        """
        self._estimator.fit(data[self._features_in], data[self._target], **kwargs)
        return self

    def _explain_using_generic_explainer(
        self,
        data: pd.DataFrame,
        algorithm: str,
        **kwargs: tp.Any,
    ) -> shap.Explanation:
        data = data[self._features_in]
        explainer = shap.Explainer(
            model=self._estimator,
            masker=data,
            algorithm=algorithm,
            **kwargs,
        )
        logger.info(f"`Using `{explainer.__class__}` to extract SHAP values...")
        return explainer(data)
