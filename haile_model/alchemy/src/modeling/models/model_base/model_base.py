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
from abc import ABC, abstractmethod
from copy import deepcopy

import pandas as pd

from ... import api
from .utils import check_columns_exist

logger = logging.getLogger(__name__)

TModel = tp.TypeVar("TModel", bound="ModelBase")
TModelFactory = tp.TypeVar("TModelFactory", bound="ModelFactoryBase")


def _check_input_len_equals_prediction_len(
    data: api.Matrix,
    prediction: api.Vector,
) -> None:
    if len(data) != len(prediction):
        raise ValueError("Length of input data is not the same as prediction length.")


class ModelBase(ABC):
    """
    Abstract class for OAI modeling
    """

    def __init__(self, features_in: tp.Iterable[str], target: str) -> None:
        self._features_in = list(features_in)
        self._target = target

    def predict(self, data: pd.DataFrame, **kwargs: tp.Any) -> api.Vector:
        """
        A public method for model prediction.

        Args:
            data: DataFrame to make a prediction
            **kwargs: Additional keyword arguments that
             might be passed for model prediction

        Returns:
            A Series or ndarray of model prediction
        """
        check_columns_exist(data, col=self._features_in)
        prediction = self._predict(data, **kwargs)
        _check_input_len_equals_prediction_len(data, prediction)
        return prediction

    def fit(self: TModel, data: pd.DataFrame, **kwargs: tp.Any) -> TModel:
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
        return self._fit(data, **kwargs)

    @property
    def features_in(self) -> tp.List[str]:
        """
        An property containing columns that are required
        to be in the input dataset for `ModelBase` `.fit` or `.predict` methods.

        Returns:
            List of column names
        """
        return self._features_in.copy()

    @property
    @abstractmethod
    def features_out(self) -> tp.List[str]:
        """
        An abstract property containing names of the features, produced as the
        result of transformations of the inputs dataset
        inside `.fit` or `.predict` methods.

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

    @abstractmethod
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

        Note:
            This method returns mapping from ``features_out`` (feature set
            that is produced after all transformations step)
            to feature importance. This is expected because
            most of the feature importance extraction technics
            utilize estimators and return importance
            for feature set used by estimator.

        Returns:
            Dict with ``features_out`` as a keys and
            feature importances as a values
        """

    @abstractmethod
    def __repr__(self) -> str:
        """
        An abstract method for string representation for models.

        Notes:
            As per definition of `__repr__` we strive to return a string representation
             that would yield an object with the same value when passed to eval();

        Returns:
            String representation.
        """

    @abstractmethod
    def _predict(self, data: pd.DataFrame, **kwargs: tp.Any) -> api.Vector:
        """
        An abstract method for model prediction logic implementation.

        Args:
            data: DataFrame to make a prediction
            **kwargs: Additional keyword arguments that
             might be passed for model prediction

        Returns:
            A Series or ndarray of model prediction
        """

    @abstractmethod
    def _fit(self: TModel, data: pd.DataFrame, **kwargs: tp.Any) -> TModel:
        """
        An abstract method for model training implementation.

         Args:
            data: DataFrame to train model on
            **kwargs: Additional keyword arguments
             that might be passed for model training

        Returns:
            A trained instance of BaseModel class
        """


class ModelFactoryBase(ABC):
    """
    Abstract class that builds ModelBase instance with `create()` method.
    """

    def __init__(
        self,
        model_init_config: tp.Dict[str, tp.Any],
        features_in: tp.Iterable[str],
        target: str,
    ) -> None:
        """
        Inheritors define the structure of `model_init_config`.

        Args:
            model_init_config: Model initialization config follows structure
             of ModelFactoryBase inheritor.
            features_in: Features column names to be used for
             model training and prediction
            target: Name of the column used in model training
        """
        self._model_init_config = model_init_config
        self._features_in = list(features_in)
        self._target = target

    @classmethod
    def from_tag_dict(
        cls: tp.Type[TModelFactory],
        model_init_config: tp.Dict[str, tp.Any],
        tag_dict: api.SupportsTagDict,
        tag_dict_features_column: str,
        target: str,
    ) -> TModelFactory:
        """
        Use this method to initialize ModelBuilder from the TagDict.
        It fetches model features and other information which potentially can be used
        for model initialization from the TagDict.

        Args:
            model_init_config: Model initialization config follows structure of
             ModelFactoryBase inheritor.
            tag_dict: Instance of TagDict with tag-level information
            tag_dict_features_column: Column name from TagDict to be used for
             identifying model features
            target: Column name to be used as model target

        Returns:
            ModelFactoryBase initialized instance
        """
        features_in = tag_dict.get_model_features(tag_dict_features_column)
        return cls(model_init_config, features_in, target)

    @property
    def model_init_config(self) -> tp.Dict[str, tp.Any]:
        return deepcopy(self._model_init_config)

    @property
    def features_in(self) -> tp.List[str]:
        return self._features_in.copy()

    @property
    def target(self) -> str:
        return self._target

    @staticmethod
    @abstractmethod
    def create_model_instance(*args: tp.Any, **kwargs: tp.Any) -> api.Estimator:
        """
        Create model instance to be wrapped with ModelBase
        """

    @abstractmethod
    def create(self) -> ModelBase:
        """
        Create `ModelBase` instance from model produced by `create_model_instance`,
        and features and target taken from TagDict.
        """

    def __repr__(self) -> str:
        """
        String representation for model factories.

        Notes:
            As per definition of `__repr__` we strive to return a string representation
             that would yield an object with the same value when passed to eval();

        Returns:
            String representation.
        """
        class_name = self.__class__.__name__
        return (
            f"{class_name}("
            f"model_init_config={self._model_init_config}, "
            f"features_in={self._features_in}, "
            f"target={self._target}"
            ")"
        )


class ModelTunerBase(ABC):
    """
    Abstract class to tune hyperparameters for `ModelBase` with `tune()` method.
    """

    def __init__(
        self,
        model_factory: ModelFactoryBase,
        model_tuner_config: tp.Dict[str, tp.Any],
    ) -> None:
        self._model_factory = model_factory
        self._model_tuner_config = model_tuner_config
        self._hyperparameters_config: tp.Optional[tp.Dict[str, tp.Any]] = None
        self._tuner_fit_kwargs: tp.Optional[tp.Any] = None

    @property
    def model_tuner_config(self) -> tp.Dict[str, tp.Any]:
        return deepcopy(self._model_tuner_config)

    @property
    def hyperparameters_config(self) -> tp.Dict[str, tp.Any]:
        return deepcopy(self._hyperparameters_config)

    def __repr__(self) -> str:
        """
        String representation for model tuners.

        Notes:
            As per definition of `__repr__` we strive to return a string representation
             that would yield an object with the same value when passed to eval();

        Returns:
            String representation.
        """
        class_name = self.__class__.__name__
        return (
            f"{class_name}("
            f"model_factory={self._model_factory}, "
            f"model_tuner_config={self._model_tuner_config}, "
            f"hyperparameters_config={self._hyperparameters_config}, "
            f"tuner_fit_kwargs={self._tuner_fit_kwargs}"
            ")"
        )

    def tune(
        self,
        data: pd.DataFrame,
        hyperparameters_config: tp.Optional[tp.Dict[str, tp.Any]] = None,
        **tuner_fit_kwargs: tp.Any,
    ) -> ModelBase:
        """
        Tune hyperparameters and return
        `ModelBase` instance with tuned hyperparameters.
        """
        self._hyperparameters_config = hyperparameters_config
        self._tuner_fit_kwargs = tuner_fit_kwargs
        return self._tune(data, hyperparameters_config, **tuner_fit_kwargs)

    @abstractmethod
    def _tune(
        self,
        data: pd.DataFrame,
        hyperparameters_config: tp.Optional[tp.Dict[str, tp.Any]],
        **tuner_fit_kwargs: tp.Any,
    ) -> ModelBase:
        """
        Abstract method for tuning hyperparameters.

        Args:
            data: data to use for model tuning
            hyperparameters_config: configuration with
             hyperparameter tuning specification
        """
