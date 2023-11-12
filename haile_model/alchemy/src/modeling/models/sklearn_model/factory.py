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

from pydantic import BaseModel

from ... import api, utils
from ..model_base import ModelFactoryBase
from .model import SklearnModel
from .target_transformer import TargetTransformerInitConfig, add_target_transformer


class SklearnModelInitConfig(BaseModel):
    estimator: utils.ObjectInitConfig
    target_transformer: tp.Optional[TargetTransformerInitConfig] = None


def load_estimator(
    model_class: str,
    model_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
) -> api.Estimator:
    """
    Loads an estimator object based on given parameters.
    Checks that loaded object has ``.fit`` and ``.predict`` methods.

    Used to load estimator objects in a Kedro pipeline.

    Args:
        model_class: string specification of modeling object to load.
        model_kwargs: dictionary of keyword arguments to the estimator class.

    Raises:
        AttributeError: if returned estimator doesn't
         have a ``.fit`` and ``.predict`` method.

    Returns:
        sklearn compatible estimator
    """
    model_kwargs = model_kwargs or {}

    estimator = utils.load_obj(model_class)(**model_kwargs)

    if getattr(estimator, "fit", None) is None:
        raise AttributeError("Model object must have a .fit method")

    if getattr(estimator, "predict", None) is None:
        raise AttributeError("Model object must have a .predict method.")

    return estimator


class SklearnModelFactory(ModelFactoryBase):
    """
    Factory class that allows creating
    ``SklearnModel`` instance based on the parametrization specified
    in ``model_init_config``.

    `model_init_config` structure should match ``SklearnModelInitConfig``::

        # Object specification for sklearn compatible estimator
        # matching the `ObjectInitConfig` structure.
        estimator:
          class_name: sklearn.linear_model.SGDRegressor
          kwargs:
            random_state: 123
            penalty: elasticnet

        # Target transform config
        # matching the `TargetTransformerInitConfig` structure.
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
             which follows structure above.
            features_in: Features column names
             to be used for model training and prediction
            target: Name of the column used in model training
        """
        super().__init__(model_init_config, features_in, target)

    @staticmethod
    def create_model_instance(init_config: tp.Dict[str, tp.Any]) -> api.Estimator:
        """
        Make ``sklearn`` estimator using ``init_config``
        """
        config = SklearnModelInitConfig(**init_config)
        estimator = load_estimator(config.estimator.class_name, config.estimator.kwargs)
        if config.target_transformer is not None:
            estimator = add_target_transformer(
                estimator,
                target_transformer=config.target_transformer.transformer,
                func=config.target_transformer.func,
                inverse_func=config.target_transformer.inverse_func,
            )
        return estimator

    def create(self) -> SklearnModel:
        """
        Build ``SklearnModel`` instance using ``create_model_instance`` method
         and ``self.model_init_config``.

        Returns:
            `SklearnModel` instance
        """
        estimator = self.create_model_instance(self._model_init_config)
        return SklearnModel(
            estimator,
            target=self._target,
            features_in=self._features_in,
        )
