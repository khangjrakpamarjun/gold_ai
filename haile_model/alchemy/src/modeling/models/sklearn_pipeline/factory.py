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

import sklearn
from pydantic import BaseModel

from ...utils import ObjectInitConfig
from ..model_base import ModelFactoryBase
from ..sklearn_model import (
    TargetTransformerInitConfig,
    add_target_transformer,
    load_estimator,
)
from .model import SklearnPipeline
from .transformer import add_transformers

WRAPPER_OPTIONS = tp.Literal["preserve_pandas", "preserve_columns", "select_columns"]


class TransformerInitConfig(BaseModel):
    class_name: str
    kwargs: tp.Dict[str, tp.Any]
    name: str
    wrapper: tp.Optional[WRAPPER_OPTIONS] = "preserve_pandas"


class SklearnPipelineInitConfig(BaseModel):
    estimator: ObjectInitConfig
    transformers: tp.List[TransformerInitConfig]
    target_transformer: tp.Optional[TargetTransformerInitConfig] = None


class SklearnPipelineFactory(ModelFactoryBase):
    """
    Factory class that allows creating
    ``SklearnPipeline`` instance based on the parametrization specified
    in ``model_init_config``.

    Structure of ``model_init_config`` that matches ``SklearnPipelineInitConfig``::

        # Object specification for sklearn compatible estimator
        # matching the `ObjectInitConfig` structure.
        estimator:
          class_name: sklearn.linear_model.SGDRegressor
          kwargs:
            random_state: 123
            penalty: elasticnet

        # List of object specification for sklearn compatible transformer
        # matching the `TransformerInitConfig` structure.
        transformers:
          - class_name: sklearn.preprocessing.StandardScaler
            kwargs: {}
            # Name of the step in sklearn.sklearn.Pipeline
            name: standard_scaler
            # `sklearn` compatible transformer do not
            # preserve column names by default.
            # We wrap it with `SklearnTransform` or `SklearnSelector`
            # classes to keep column names during transformations.
            # Use flag options below to choose which wrapper to use:
            # `wrapper: select_columns` to wrap with `SkLearnSelector`
            # `wrapper: preserve_columns` to wrap with `SklearnTransform`
            # `wrapper: preserve_pandas` to wrap with
            #  `ColumnNamesAsNumbers` (default option)
            # `wrapper: null` to skip wrapping
            wrapper: preserve_columns

        # Target transform config matching the
        `TargetTransformerInitConfig` structure.
        # Either transformer key should be filled,
        # or `func` and `inverse_func` keys should be filled.
        target_transformer:
          # Object specification for transformer
          # matching the `ObjectInitConfig` structure
          transformer: null
          # Path for the function to use as target transform
          func: numpy.log1p
          # Path for the function to use as a inverse target transform
          inverse_func: numpy.expm1
    """

    def __init__(  # noqa: WPS612
        self,
        model_init_config: tp.Dict[str, tp.Any],
        features_in: tp.Iterable[str],
        target: str,
    ) -> None:
        """
        Args:
            model_init_config: Model initialization config
             which the structure shown above.
            features_in: Features column names
             to be used for model training and prediction
            target: Name of the column used in model training
        """
        super().__init__(model_init_config, features_in, target)

    @staticmethod
    def create_model_instance(
        init_config: SklearnPipelineInitConfig,
    ) -> sklearn.pipeline.Pipeline:
        """
        Make ``sklearn.pipeline.Pipeline`` instance using ``init_config``
        """
        estimator = load_estimator(
            init_config.estimator.class_name,
            init_config.estimator.kwargs,
        )
        if init_config.target_transformer is not None:
            estimator = add_target_transformer(
                estimator=estimator,
                target_transformer=init_config.target_transformer.transformer,
                func=init_config.target_transformer.func,
                inverse_func=init_config.target_transformer.inverse_func,
            )
        return add_transformers(
            estimator=estimator,
            transformers=[
                transformer_config.dict()
                for transformer_config in init_config.transformers
            ],
        )

    def create(self) -> SklearnPipeline:
        """
        Build ``SklearnPipeline`` instance.

        Returns:
            ``SklearnPipeline`` instance
        """
        config = SklearnPipelineInitConfig(**self._model_init_config)
        pipeline = self.create_model_instance(config)
        return SklearnPipeline(
            pipeline,
            target=self._target,
            features_in=self._features_in,
        )
