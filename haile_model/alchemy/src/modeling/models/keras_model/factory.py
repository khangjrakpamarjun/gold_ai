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
from abc import ABC, abstractmethod

from ..model_base import ModelFactoryBase
from .errors import get_tensorflow_error_message
from .model import KerasModel

try:
    from tensorflow import keras
except ImportError as import_error:
    raise ImportError(get_tensorflow_error_message()) from import_error


class KerasModelFactory(ModelFactoryBase, ABC):
    """
    Abstract class to build ``KerasModel`` instance using ``create()`` method.
    """

    @staticmethod
    @abstractmethod
    def create_model_instance(*args: tp.Any, **kwargs: tp.Any) -> keras.Model:
        """
        Abstract method to create custom ``tensorflow.keras.Model`` instance
        from parameters specified in function definition.

        This method should be overwritten by the inheritor of ``KerasModelFactory``
        with the custom ``KerasModel`` initialization logic.
        ``self.model_init_config`` dictionary is being
        unwrapped into this function input.

        See example implementation below.

        Example implementation::

            class UserDefinedKerasModelFactory(KerasModelFactory):
            @staticmethod
            def create_model_instance(
                units: int = 32,
                learning_rate: float = 1e-03,
            ) -> keras.Model:
                model = keras.Sequential(
                    [
                        keras.layers.Normalization(axis=-1),
                        keras.layers.Dense(units=units, activation="tanh"),
                        keras.layers.Dense(units=1),
                    ]
                )
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                    loss="mean_squared_error",
                    metrics=[
                        keras.metrics.MeanAbsoluteError(),
                        keras.metrics.MeanSquaredError(),
                        keras.metrics.MeanAbsolutePercentageError(),
                    ],
                )
                return model

        """

    def create(self) -> KerasModel:
        """
        Create ``KerasModel`` instance from model produced by ``create_model_instance``.

        Returns:
            ``KerasModel`` instance
        """
        model = self.create_model_instance(**self._model_init_config)
        model.build((None, len(self._features_in)))
        return KerasModel(model, self._features_in, self._target)
