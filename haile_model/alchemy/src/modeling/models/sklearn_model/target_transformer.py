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
from sklearn.compose import TransformedTargetRegressor

from ... import utils


class TargetTransformerInitConfig(BaseModel):
    transformer: tp.Optional[utils.ObjectInitConfig] = None
    func: tp.Optional[str] = None
    inverse_func: tp.Optional[str] = None


def add_target_transformer(
    estimator: tp.Any,
    target_transformer: tp.Optional[utils.ObjectInitConfig] = None,
    func: tp.Optional[str] = None,
    inverse_func: tp.Optional[str] = None,
) -> TransformedTargetRegressor:
    """Wraps the given estimator in a TransformedTargetRegressor.

    TransformedTargetRegressor can be initialized using the transformer init config
    or using the pair of a ``func`` and ``inverse_func``.


    See documentation for ``TransformedTargetRegressor``:
        scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html

    Examples:

    Using the `target_transformer` object init config::

        target_transformer = {
            "class_name": "sklearn.preprocessing.MinMaxScaler",
            "kwargs": {
                "feature_range": [0,2],
            }
        }
        add_target_transformer(
            LinearRegression(),
            target_transformer=ObjectInitConfig(**target_transformer),
        )

    Using `func` and `inverse_func` pair::

        add_target_transformer(
            LinearRegression(),
            func="numpy.log1p",
            inverse_func="numpy.expm1",
        )

    Returns:
        ``sklearn.compose.TransformedTargetRegressor`` with wrapped estimator
    """
    if target_transformer is not None:
        transformer = utils.load_obj(target_transformer.class_name)(
            **target_transformer.kwargs,
        )
        return TransformedTargetRegressor(regressor=estimator, transformer=transformer)

    if func is not None and inverse_func is not None:
        func = utils.load_obj(func)
        inverse_func = utils.load_obj(inverse_func)
        return TransformedTargetRegressor(
            regressor=estimator,
            func=func,
            inverse_func=inverse_func,
        )
    raise ValueError(
        "Target transformer can't be initialised with provided "
        "target transformer init config is not",
    )
