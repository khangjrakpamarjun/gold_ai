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

import numpy as np
import pandas as pd
import shap

from ..metrics_utils import evaluate_regression_metrics
from ..model_base import EvaluatesMetrics, ModelBase, ProducesShapFeatureImportance
from .errors import get_tensorflow_error_message

try:
    from tensorflow import keras
except ImportError as error:
    raise ImportError(get_tensorflow_error_message()) from error

logger = logging.getLogger(__name__)


def ensure_data_index_increasing(data: pd.DataFrame) -> None:
    if not data.index.is_monotonic_increasing:
        logging.warning(
            "Data index is expected to be increasing."
            " Sorting dataset to ensure"
            " data index is sorted properly.",
        )
        data.sort_index(inplace=True)


class KerasModel(
    ModelBase,
    ProducesShapFeatureImportance,
    EvaluatesMetrics,
):
    """
    A wrapper for ``tensorflow.keras`` models with OAI related functionality.
    """

    _DEFAULT_FIT_PARAMS: tp.Dict[str, tp.Any] = {
        "early_stopping_metric": "val_loss",
        "early_stopping_patience": 10,
        "validation_split": 0.2,
        "epochs": 300,
        "verbose": 1,
    }

    def __init__(
        self,
        keras_model: keras.Model,
        features_in: tp.Iterable[str],
        target: str,
    ) -> None:
        """
        Args:
            keras_model: `tensorflow.keras.Model` instance to wrap
            target: Name of the column used in model training
            features_in: A list of feature names that are used
             to select features for model training or prediction.
        """
        super().__init__(features_in=features_in, target=target)
        input_shape = (None, len(self._features_in))
        keras_model.build(input_shape)
        self._keras_model = keras_model

    @property
    def features_out(self) -> tp.List[str]:
        return self.features_in

    @property
    def keras_model(self) -> keras.Model:
        return self._keras_model

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
            data: data to calculate metrics on
            **kwargs: keyword arguments passed to `.predict` method

        Returns:
            Dictionary from metric name into metric value
        """
        target = data[self._target].values
        prediction = self.predict(data, **kwargs)
        return evaluate_regression_metrics(target, prediction)

    def get_feature_importance(
        self,
        data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> tp.Dict[str, float]:
        return self.get_shap_feature_importance(
            data[self._features_in],
            **kwargs,
        )

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        json_representation = self._keras_model.to_json()
        return (
            f"{class_name}("
            f"keras_model={json_representation}, "
            f'target="{self._target}" ,'
            f"features_in={self._features_in}"
            ")"
        )

    def _produce_shap_explanation(
        self,
        data: pd.DataFrame,
        algorithm: str = "deep",
        **kwargs: tp.Any,
    ) -> shap.Explanation:
        """
        Produce an instance of shap.Explanation based on provided data.

        Args:
            data: data to extract SHAP values from
            algorithm: algorithm to use;
             `algorithm='deep'` utilizes legacy `shap.DeepExplainer`,
             other potential values of this
             argument are passed into generic `shap.Explainer`
            **kwargs: keyword arguments to be provided
              for `shap.Explainer` initialization

        Returns:
           `shap.Explanation` for ``self.features_in`` feature set
            containing prediction base values and SHAP values
        """
        if algorithm == "deep":
            return self._explain_using_deep_explainer(data)

        return self._explain_using_generic_explainer(
            data=data,
            algorithm=algorithm,
            **kwargs,
        )

    def _fit(self, data: pd.DataFrame, **kwargs: tp.Any) -> KerasModel:
        """
        Train wrapped tensorflow.keras.Model
        Args:
            data: DataFrame, that contains `self.features_in`
             feature set to train model
            **kwargs: Key-word arguments to be passed to
             ``tensorflow.keras.Model`` method ``.fit``

        Notes:
            Training data should be sorted by index. Early stopping is
            used in order to prevent models overfitting.
            Last `validation_split * len(data)` samples
            are used as a validation dataset.
            Dataset should be sorted by index to prevent data leakage.

        Returns:
            A trained instance of ``KerasModel``
        """
        ensure_data_index_increasing(data)
        modified_kwargs = self._modify_fit_kwargs(kwargs)
        self._keras_model.fit(
            data[self._features_in],
            data[self._target],
            **modified_kwargs,
        )
        return self

    def _predict(self, data: pd.DataFrame, **kwargs: tp.Any) -> np.array:
        """
        Make a prediction using wrapped ``tensorflow.keras.Model``.

        Args:
            data: DataFrame, that contains `self.features_in`
             feature set to make predictions
            **kwargs: Key-word arguments to be passed to `sklearn` method `.predict`

        Returns:
            A Series or ndarray with prediction.
        """
        return self._keras_model.predict(data[self._features_in], **kwargs).squeeze()

    def _modify_fit_kwargs(self, kwargs: tp.Any) -> tp.Dict[str, tp.Any]:
        validation_split = (
            kwargs.pop("validation_split", None)
            or self._DEFAULT_FIT_PARAMS["validation_split"]
        )
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=(
                kwargs.pop("early_stopping_metric", None)
                or self._DEFAULT_FIT_PARAMS["early_stopping_metric"]
            ),
            patience=(
                kwargs.pop("early_stopping_patience", None)
                or self._DEFAULT_FIT_PARAMS["early_stopping_patience"]
            ),
        )
        verbose = (
            kwargs.pop("verbose")
            if "verbose" in kwargs
            else self._DEFAULT_FIT_PARAMS["verbose"]
        )
        epochs = kwargs.pop("epochs", None) or self._DEFAULT_FIT_PARAMS["epochs"]
        return dict(
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=verbose,
            epochs=epochs,
            **kwargs,
        )

    def _explain_using_deep_explainer(self, data: pd.DataFrame) -> shap.Explanation:
        explainer = shap.DeepExplainer(
            model=self._keras_model,
            data=data[self._features_in].values,
        )
        logger.info(f"`Using `{explainer.__class__}` to extract SHAP values...")
        shap_values, *_ = explainer.shap_values(data[self._features_in].values)
        return shap.Explanation(
            values=shap_values,
            data=data[self._features_in].values,
            base_values=np.repeat(explainer.expected_value.numpy(), len(data)),
            feature_names=self._features_in,
        )

    def _explain_using_generic_explainer(
        self,
        data: pd.DataFrame,
        algorithm: str,
        **kwargs: tp.Any,
    ) -> shap.Explanation:
        explainer = shap.Explainer(
            model=self._keras_model,
            masker=data[self._features_in],
            algorithm=algorithm,
            **kwargs,
        )
        logger.info(f"`Using `{explainer.__class__}` to extract SHAP values...")
        return explainer(data[self._features_in])
