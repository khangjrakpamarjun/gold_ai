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
"""
Functions for loading objects from config dictionaries.
"""

import typing as tp

from ... import api, utils
from ..model_base import ModelBase, ModelFactoryBase

TModel = tp.TypeVar("TModel", bound=ModelBase)
TModelFactory = tp.TypeVar("TModelFactory", bound=ModelFactoryBase)


def create_model_factory_from_tag_dict(
    model_factory_type: tp.Union[str, tp.Type[TModelFactory]],
    model_init_config: tp.Dict[str, tp.Any],
    tag_dict: api.SupportsTagDict,
    tag_dict_features_column: str,
    target: str,
) -> TModelFactory:
    """
    Create ``ModelFactoryBase`` instance from type and model initialization config.
    This function is mostly needed for pipelining.

    Each ``model_factory_type`` requires special structure
     for ``model_init_config`` dict, see corresponding Factory for more details.

    Args:
        model_factory_type: ``ModelFactoryBase`` type that
         has ``.create`` method that created ``ModelBase`` instance.
        model_init_config: Dict with parameters required
         for ``ModelFactoryBase`` initialization.
        target: Column name to be used as a target in model
        tag_dict: TagDict instance to select features for model
        tag_dict_features_column: Column name in the ``TagDict``
         to select features for model as ``features_in``

    Returns:
        Instance of a ``ModelBase`` inheritor class specified by ``model_type``
    """
    if isinstance(model_factory_type, str):
        model_factory_type = utils.load_obj(model_factory_type)
    return model_factory_type.from_tag_dict(
        model_init_config=model_init_config,
        tag_dict=tag_dict,
        tag_dict_features_column=tag_dict_features_column,
        target=target,
    )


def create_model_factory(
    model_factory_type: tp.Union[str, tp.Type[TModelFactory]],
    model_init_config: tp.Dict[str, tp.Any],
    features: tp.List[str],
    target: str,
) -> TModelFactory:
    """
    Create ``ModelFactoryBase`` instance from type and model initialization config.
    This function is mostly needed for pipelining.

    Each ``model_factory_type`` requires special structure
    for ``model_init_config`` dict,
    see corresponding Builders for more details.

    Args:
        model_factory_type: ``ModelFactoryBase`` type
         that has ``.create`` method that created ModelBase instance.
        model_init_config: Dict with parameters
         required for ``ModelFactoryBase`` initialization.
        features: List of features used in the model as ``features_in``
        target: Column name to be used as a target in model

    Returns:
        Instance of a ``ModelBase`` inheritor class specified by ``model_type``
    """
    if isinstance(model_factory_type, str):
        model_factory_type = utils.load_obj(model_factory_type)
    return model_factory_type(model_init_config, features, target)


def create_model(model_factory: TModelFactory) -> TModel:
    """
    Create model from the ``ModelFactoryBase``'s ``.create()`` method.
    This function is mostly needed for pipelining.
    Args:
        model_factory: Instance of ``ModelFactoryBase``

    Returns:
        Instance of ``ModelBase`` produced by ``ModelFactoryBase``
    """
    return model_factory.create()
