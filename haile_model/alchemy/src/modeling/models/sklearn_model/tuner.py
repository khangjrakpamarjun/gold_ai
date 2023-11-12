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
import logging
import typing as tp
from ast import literal_eval

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, mean_absolute_percentage_error
from sklearn.model_selection._search import BaseSearchCV  # noqa: WPS436

from ... import utils
from ..model_base import ModelTunerBase
from .factory import SklearnModelFactory
from .model import SklearnModel

logger = logging.getLogger(__name__)


class SklearnModelTuner(ModelTunerBase):
    """
    A class to tune hyperparameters for `SklearnModel` instance.
    """

    def __init__(
        self,
        model_factory: SklearnModelFactory,
        model_tuner_config: tp.Dict[str, tp.Any],
    ) -> None:
        """
        model_tuner_config structure should match `SklearnTunerConfig` structure::

            # Object specification for sklearn compatible CV tuner
            # matching the `ObjectInitConfig` structure.
            init:
                class_name: sklearn.model_selection.GridSearchCV
                kwargs:
                  n_jobs: -1
                  refit: mae
                  param_grid:
                    estimator__alpha: [0.0001, 0.001, 0.01, 0.1, 1, 10]
                    estimator__l1_ratio: [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
                  scoring:
                    mae: neg_mean_absolute_error
                    rmse: neg_root_mean_squared_error
                    r2: r2

        Args:
            model_factory: Builder instance that produces model with corresponding type
            model_tuner_config: Dictionary with the structure defined above
        """
        super().__init__(model_factory, model_tuner_config)
        self._tuner: tp.Optional[BaseSearchCV] = None

    @property
    def tuner(self) -> BaseSearchCV:
        return copy.copy(self._tuner)

    def _tune(
        self,
        data: pd.DataFrame,
        hyperparameters_config: tp.Optional[tp.Dict[str, tp.Any]] = None,
        **tuner_fit_kwargs: tp.Any,
    ) -> SklearnModel:
        """
        Tune model for model using the strategy defined in

        Args:
            data: data to use for hyperparameters tuning
            hyperparameters_config: Dict with hyperparameters tuning configuration.
             Not used in this class.
            **tuner_fit_kwargs: keyword arguments for `.fit` method.

        Returns:
            tuned SklearnModel instance
        """
        tuner_config = utils.ObjectInitConfig(**self._model_tuner_config)
        logger.info("Initializing sklearn hyperparameters tuner...")
        sklearn_model = self._model_factory.create()
        self._tuner = initialize_sklearn_hyperparameters_tuner(
            tuner_config.dict(),
            sklearn_model.estimator,
        )
        logger.info("Tuning hyperparameters...")
        self._tuner.fit(
            data[sklearn_model.features_in],
            data[sklearn_model.target],
            **tuner_fit_kwargs,
        )
        return SklearnModel(
            self._tuner.best_estimator_,
            features_in=sklearn_model.features_in,
            target=sklearn_model.target,
        )


def initialize_sklearn_hyperparameters_tuner(  # noqa: WPS231
    tuner_config: tp.Dict[str, tp.Any],
    estimator: BaseEstimator,
) -> BaseSearchCV:
    """Instantiates a hyperparameters tuning strategy from `hyperparameters_config`.
    `hyperparameters_config` stands for a dict that matches the ObjectInitConfig format.

    If scoring is not specified, MAPE is used.

    Example tuner hyperparameters_config for a simple
    ``sklearn.neural_network.MLPRegressor``::

        class_name: sklearn.model_selection.GridSearchCV
        kwargs:
            n_jobs: -1
            refit: mae
            cv: 3
            param_grid:
                estimator__hidden_layer_sizes: [ [ 15 ], [ 15,3 ] ]
                estimator__learning_rate: [ 'constant', 'adaptive' ]
                estimator__activation: [ 'identity', 'relu' ]
            scoring:
                mae: neg_mean_absolute_error
                rmse: neg_root_mean_squared_error
                r2: r2

    This example assumes the estimator is inside an sklearn Pipeline. If this is not the
    case, simple remove the ``estimator__`` prefixes.

    If loading a more complicated parameter distribution, use the syntax below::

        param_distribution:
            estimator__hidden_layer_size: [ [ 15 ], [ 15,3 ] ]
            estimator__some_fancy_parameter:
                class_name: scipy.stats.norm
                kwargs:
                    loc: 10
                    scale: 0.5

    Args:
        tuner_config: dictionary of parameters mathing the ObjectInitConfig.
        estimator: an object conforming to the sklearn API.

    Returns:
        Object conforming to the sklearn BaseSearchCV API.
    """
    tuner_kwargs = tuner_config["kwargs"].copy()
    tuner_kwargs["estimator"] = estimator

    if "cv" in tuner_kwargs:
        cv = tuner_kwargs["cv"]

        if not isinstance(cv, int):
            cv = utils.load_obj(tuner_kwargs["cv"]["class"])(
                **tuner_kwargs["cv"]["kwargs"],
            )

        tuner_kwargs["cv"] = cv

    if "param_distributions" in tuner_kwargs:
        parameters_distribution = tuner_kwargs["param_distributions"]
        for parameter_name, parameter_distribution in parameters_distribution.items():
            if isinstance(parameter_distribution, str):
                parameters_distribution[parameter_name] = literal_eval(
                    parameter_distribution
                )  # noqa: E501

            elif isinstance(parameter_distribution, dict):
                parameters_distribution[parameter_name] = utils.load_obj(
                    parameters_distribution[parameter_name]["class"],
                )(**parameters_distribution[parameter_name]["kwargs"])

    if "scoring" not in tuner_kwargs:
        tuner_kwargs["scoring"] = {"mape": None}

    if "mape" in tuner_kwargs["scoring"]:
        tuner_kwargs["scoring"]["mape"] = make_scorer(
            mean_absolute_percentage_error,
            greater_is_better=False,
        )

    return utils.load_obj(tuner_config["class_name"])(**tuner_kwargs)
