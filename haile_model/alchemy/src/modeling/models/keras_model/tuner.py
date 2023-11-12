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

import copy
import typing as tp
from abc import ABC, abstractmethod

import pandas as pd
from pydantic import BaseModel

from optimus_core import partial_wrapper

from ... import utils
from ..model_base import ModelTunerBase
from .errors import get_tensorflow_error_message
from .factory import KerasModelFactory
from .model import KerasModel, ensure_data_index_increasing

try:  # noqa: WPS229
    import keras_tuner
    from tensorflow import keras
except ImportError as error:
    raise ImportError(get_tensorflow_error_message()) from error


class KerasModelTunerConfig(BaseModel):
    tuner: str
    objective_metric: str
    objective_direction: str
    max_trials: tp.Optional[int] = None
    executions_per_trial: tp.Optional[int] = None
    project_name: tp.Optional[str] = None
    tuning_artefacts_dir: tp.Optional[str] = None


class KerasModelTuner(ModelTunerBase, ABC):
    """
    A class to tune hyperparameters for ``KerasModel`` instance.
    """

    _RANDOM_SEED: int = 0
    _DEFAULT_TUNING_PARAMS: tp.Dict[str, tp.Union[str, float]] = {
        "max_trials": 25,
        "early_stopping_metric": "val_loss",
        "directory": "data/sample_keras_model_hp",
        "project_name": "oai-keras-model",
        "early_stopping_patience": 10,
        "executions_per_trial": 1,
        "validation_share": 0.2,
        "epochs": 300,
        "verbose": 1,
    }

    def __init__(
        self,
        model_factory: KerasModelFactory,
        model_tuner_config: tp.Dict[str, tp.Any],
    ) -> None:
        """
        Structure of ``model_tuner_config`` should be
        aligned with ``KerasModelTunerConfig``::

            # Path to instance of tuner in keras_tuner
            tuner: keras_tuner.RandomSearch
            # Metric for hyperparameters objective and direction
            objective_metric: mean_squared_error
            objective_direction: min
            # Number of hyperparameter tuning trials and executions per trial
            max_trials: 5
            executions_per_trial: 1
            # Project name
            project_name: oai-keras-model
            # Path to the directory to keep training artefacts
            tuning_artefacts_dir: data/sample_keras_model_hp

        Args:
            model_factory: instance of the KerasModelFactory
            model_tuner_config: configuration dictionary
             with the structure defined above
        """
        super().__init__(model_factory, model_tuner_config)
        self._tuner: tp.Optional[keras_tuner.Tuner] = None

    @property
    def tuner(self) -> keras_tuner.Tuner:
        return copy.copy(self._tuner)

    def _tune(
        self,
        data: pd.DataFrame,
        hyperparameters_config: tp.Optional[tp.Dict[str, tp.Any]],
        **tuner_fit_kwargs: tp.Any,
    ) -> KerasModel:
        """
        Tune hyperparameters using the tuning strategy
         defined in ``_create_trial_hyperparameters()``
         and model initialization rules defined in
         `KerasModelFactory.create_model_instance()`.
        Args:
            data: DataFrame to tune hyperparameters on.
            hyperparameters_config: Hyperparameter tuning
             rules as required in ``_create_trial_hyperparameters()``
            **tuner_fit_kwargs: Key-word arguments to be passed in model ``.fit`` method

        Returns:
            KerasModel instance with tuned hyperparameters
        """
        ensure_data_index_increasing(data)
        tuner_config = KerasModelTunerConfig(**self.model_tuner_config)
        tuner_keyword_args = self._extract_kw_args_for_keras_tuner_init(
            tuner_config,
            hyperparameters_config,
        )
        self._tuner = utils.load_obj(tuner_config.tuner)(**tuner_keyword_args)
        tuner_fit_kwargs = self._extract_kw_args_for_keras_tuner_search(
            tuner_fit_kwargs,
        )
        self._tuner.search(
            data[self._model_factory.features_in],
            data[self._model_factory.target],
            **tuner_fit_kwargs,
        )
        best_model, *_ = self._tuner.get_best_models()
        return KerasModel(
            best_model,
            self._model_factory.features_in,
            self._model_factory.target,
        )

    @staticmethod
    @abstractmethod
    def _create_trial_hyperparameters(
        hp: keras_tuner.HyperParameters,
        model_init_config: tp.Dict[str, tp.Any],
        model_hyperparameters_config: tp.Dict[str, tp.Any],
    ) -> tp.Dict[str, tp.Any]:
        """
        Abstract method to specify hyperparameters tuning strategy
        for `tensorflow.keras.Model` instance using ``keras_tuner.HyperParameters``.

        This method should be overwritten by the inheritor of ``KerasModelTuner``
        with the custom hyperparameters tuning logic. See example implementation below.

        Example implementation::

            class UserKerasModelTuner(KerasModelTuner):
                @staticmethod
                def _create_trial_hyperparameters(
                    #  Hyperparameter tuning API is based on `keras_tuner` package.
                    hp: keras_tuner.HyperParameters,
                    #  You also might want to use hyperparameters settings from Factory,
                    #  those are available in `model_init_config` dictionary.
                    model_init_config: tp.Dict[str, tp.Any],
                    #  Use `model_hyperparameters_config` to specify strategy.
                    #  See the example below.
                    model_hyperparameters_config: tp.Dict[str, tp.Any],
                ) -> tp.Dict[str, tp.Any]:
                    units = hp.Int(
                        "units",
                        min_value=model_hyperparameters_config["units"]["min_value"],
                        max_value=model_hyperparameters_config["units"]["max_value"],
                        sampling=model_hyperparameters_config["units"]["sampling"],
                    )
                    learning_rate = model_init_config["learning_rate"]
                    return {"units": units, "learning_rate": learning_rate}

        Args:
            hp: instance to tune the ``keras_tuner.HyperParameters``
            model_init_config: model initialization config
             used for ``KerasModelFactory``
            model_hyperparameters_config: dictionary with
             hyperparameters tuning specification

        Returns:
            Dictionary mapping hyperparameter name to trial hyperparameter value
        """

    def _create_trial_model(
        self,
        hp: keras_tuner.HyperParameters,
        model_hyperparameters_config: tp.Dict[str, tp.Any],
    ) -> keras.Model:
        trial_hyperparameters = self._create_trial_hyperparameters(
            hp=hp,
            model_init_config=self._model_factory.model_init_config,
            model_hyperparameters_config=model_hyperparameters_config,
        )
        return self._model_factory.create_model_instance(**trial_hyperparameters)

    def _extract_kw_args_for_keras_tuner_init(
        self,
        tuner_config: KerasModelTunerConfig,
        hyperparameters_config: tp.Dict[str, tp.Any],
    ) -> tp.Dict[str, tp.Any]:
        hypermodel = partial_wrapper(
            self._create_trial_model,
            model_hyperparameters_config=hyperparameters_config,
        )
        objective = keras_tuner.Objective(
            name=tuner_config.objective_metric,
            direction=tuner_config.objective_direction,
        )
        max_trials = (
            tuner_config.max_trials or self._DEFAULT_TUNING_PARAMS["max_trials"]
        )
        executions_per_trial = (
            tuner_config.executions_per_trial
            or self._DEFAULT_TUNING_PARAMS["executions_per_trial"]
        )
        directory = (
            tuner_config.tuning_artefacts_dir
            or self._DEFAULT_TUNING_PARAMS["directory"]
        )
        project_name = (
            tuner_config.project_name or self._DEFAULT_TUNING_PARAMS["project_name"]
        )
        return {
            "hypermodel": hypermodel,
            "objective": objective,
            "max_trials": max_trials,
            "executions_per_trial": executions_per_trial,
            "overwrite": True,
            "seed": self._RANDOM_SEED,
            "directory": directory,
            "project_name": project_name,
        }

    def _extract_kw_args_for_keras_tuner_search(
        self,
        tuner_fit_kwargs: tp.Dict[str, tp.Any],
    ) -> tp.Dict[str, tp.Any]:
        validation_split = (
            tuner_fit_kwargs.pop("validation_share", None)
            or self._DEFAULT_TUNING_PARAMS["validation_share"]
        )
        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor=(
                tuner_fit_kwargs.pop("early_stopping_metric", None)
                or self._DEFAULT_TUNING_PARAMS["early_stopping_metric"]
            ),
            patience=(
                tuner_fit_kwargs.pop("early_stopping_patience", None)
                or self._DEFAULT_TUNING_PARAMS["early_stopping_patience"]
            ),
        )
        verbose = (
            tuner_fit_kwargs.pop("verbose", None)
            or self._DEFAULT_TUNING_PARAMS["verbose"]
        )
        epochs = (
            tuner_fit_kwargs.pop("epochs", None)
            or self._DEFAULT_TUNING_PARAMS["epochs"]
        )
        return dict(
            validation_split=validation_split,
            callbacks=[early_stopping_callback],
            verbose=verbose,
            epochs=epochs,
            **tuner_fit_kwargs,
        )
