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

import pandas as pd

from ... import utils
from ..model_base import ModelBase, ModelFactoryBase, ModelTunerBase

TModel = tp.TypeVar("TModel", bound=ModelBase)
TModelFactory = tp.TypeVar("TModelFactory", bound=ModelFactoryBase)
TModelTuner = tp.TypeVar("TModelTuner", bound=ModelTunerBase)


def create_tuner(
    model_factory: ModelFactoryBase,
    model_tuner_type: tp.Union[str, tp.Type[TModelTuner]],
    tuner_config: tp.Dict[str, tp.Any],
) -> TModelTuner:
    """
    Create model tuner instance

    Args:
        model_factory: Instance of ``ModelFactoryBase``
         produces Model with corresponding type
        model_tuner_type: class name of inheritor of
         ``ModelTunerBase`` to be initialized
        tuner_config: a dict for hyperparameters
         tuning config aligned with provided model

    Returns:
        ``TunerBase`` instance
    """
    if isinstance(model_tuner_type, str):
        model_tuner_type = utils.load_obj(model_tuner_type)
    return model_tuner_type(model_factory, tuner_config)


def tune_model(
    model_tuner: TModelTuner,
    hyperparameters_config: tp.Optional[tp.Dict[str, tp.Any]],
    data: pd.DataFrame,
    **tuner_fit_kwargs: tp.Any,
) -> TModel:
    """
    Tune hyperparameters for instance of ``ModelBase`` using model tuner
    and tuner parameters in ``hyperparameters_config``.

    Args:
        model_tuner: model tuner instance
        hyperparameters_config: config with hyperparameter tuning specification
        data: data to use for hyperparameters tuning
        **tuner_fit_kwargs: keyword arguments to ``.fit`` method.

    Returns:
        ``BaseModel`` instance with hyperparameters hyperparameters tuned
        and DataFrame with tuning results.
    """
    return model_tuner.tune(
        hyperparameters_config=hyperparameters_config,
        data=data,
        **tuner_fit_kwargs,
    )
