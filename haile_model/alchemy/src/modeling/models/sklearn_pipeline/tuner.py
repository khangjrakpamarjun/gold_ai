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

import pandas as pd
import sklearn
from sklearn.model_selection._search import BaseSearchCV  # noqa: WPS436

from ... import utils
from ..model_base import ModelTunerBase
from ..sklearn_model import initialize_sklearn_hyperparameters_tuner
from .factory import SklearnPipelineFactory
from .model import SklearnPipeline

logger = logging.getLogger(__name__)


class SklearnPipelineTuner(ModelTunerBase):
    """
    A class to tune hyperparameters for `SklearnPipeline` instance.
    """

    def __init__(
        self,
        model_factory: SklearnPipelineFactory,
        model_tuner_config: tp.Dict[str, tp.Any],
    ) -> None:
        """
        tuner_config structure should match `SklearnTunerConfig` structure::

            # Object specification for sklearn compatible CV tuner
            # matching the `ObjectInitConfig` structure.
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
        """
        super().__init__(model_factory, model_tuner_config)
        self._model_factory = model_factory
        self._tuner: tp.Optional[BaseSearchCV] = None

    @property
    def tuner(self) -> BaseSearchCV:
        return copy.copy(self._tuner)

    def _tune(
        self,
        data: pd.DataFrame,
        hyperparameters_config: tp.Optional[tp.Dict[str, tp.Any]] = None,
        **tuner_fit_kwargs: tp.Any,
    ) -> SklearnPipeline:
        """
        Args:
            data: data to use for hyperparameters tuning
            hyperparameters_config: Dict with hyperparameters tuning
             configuration. Not used on this class.
            **tuner_fit_kwargs: keyword arguments for `.fit` method.

        Returns:
            tuned SklearnPipeline instance
        """
        tuner_config = utils.ObjectInitConfig(**self._model_tuner_config)
        sklearn_pipeline = self._model_factory.create().fit(data)
        logger.info("Initializing sklearn hyperparameters tuner...")
        self._tuner = initialize_sklearn_hyperparameters_tuner(
            tuner_config.dict(),
            sklearn_pipeline.get_pipeline(),
        )
        logger.info("Tuning hyperparameters...")
        self._tuner.fit(data, data[sklearn_pipeline.target], **tuner_fit_kwargs)
        selector, *pipeline_steps = self._tuner.best_estimator_.steps
        return SklearnPipeline(
            sklearn.pipeline.Pipeline(pipeline_steps),
            features_in=sklearn_pipeline.features_in,
            features_out=sklearn_pipeline.features_out,
            target=sklearn_pipeline.target,
        )
