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
import warnings
from enum import Enum

import sklearn

from optimus_core import (
    ColumnNamesAsNumbers,
    SkLearnSelector,
    SklearnTransform,
    Transformer,
)

from ... import utils

TTransformerConfig = tp.Union[tp.Tuple[str, tp.Any], tp.Dict[str, tp.Any]]
TTransformerStep = tp.Union[tp.Dict[str, tp.Any], tp.Tuple[str, Transformer]]


class WrapperOptions(Enum):
    SELECT_COLUMNS: str = "select_columns"
    PRESERVE_COLUMNS: str = "preserve_columns"
    PRESERVE_PANDAS: str = "preserve_pandas"


def load_transformer(  # noqa: WPS210,WPS231
    transformer_config: TTransformerConfig,
) -> tp.Tuple[str, Transformer]:
    """Instantiates a transformer object if one is provided.

    Transformer config expects the following
     format that matches `TransformerInitConfig`:
        - class_name: full import path of class
         (e.g. sklearn.feature_selection.SelectKBest)
        - kwargs: keyword argument to pass to class constructor.
        - name: string name to use as the name of the step in the Sklearn pipeline.
        - wrapper: Optional wrapper to use to preserve ``pd.DataFrame``
         for output of the transformer. If wrapper is set to:
             - ``select_columns``,
              wraps transformer with ``optimus_core.SkLearnSelector``
             - ``preserve_columns``,
              wraps transformer with ``optimus_core.SklearnTransform``
             - ``preserve_pandas``,
              wraps transformer with ``ColumnNamesAsNumbers``
             - ``None``, transformer is not wrapped.

    Returns:
        (transformer name, transformer object)
    """
    # If we were provided a transformer already, just return it.
    # TODO: refactor multiline condition
    if (  # noqa: WPS337
        len(transformer_config) == 2
        and not isinstance(transformer_config, dict)
        and getattr(transformer_config[1], "fit", None) is not None
    ):
        transformer = transformer_config[1]
        if not isinstance(transformer, Transformer):
            transformer_type = type(transformer)
            warnings.warn(
                f"Got a transformer of type {transformer_type} which may not "
                "preserve column names when using a pandas DataFrame. "
                "Wrap the transformer in an "
                "optimus_core.transformer.SklearnSelector or "
                "optimus_core.transformer.SklearnTransformer to retain columns.",
                RuntimeWarning,
            )

        return transformer_config
    required_keys = ["class_name", "kwargs", "name", "wrapper"]
    if not all(key in transformer_config for key in required_keys):
        missing = list(set(required_keys) - set(transformer_config.keys()))
        raise KeyError(
            f"Your transformer_config {transformer_config} is missing"
            f" the following keys {missing}.",
        )

    class_name = transformer_config.get("class_name")
    kwargs = transformer_config.get("kwargs", {})
    step_name = transformer_config.get("name", "transformer_step")

    if "estimator" in kwargs:
        estimator_kwargs = kwargs["estimator"].pop("kwargs", {})
        estimator_class = kwargs["estimator"].pop("class", "")
        kwargs["estimator"] = utils.load_obj(estimator_class)(**estimator_kwargs)

    if "score_func" in kwargs:
        kwargs["score_func"] = utils.load_obj(kwargs["score_func"])

    sklearn_transformer = class_name
    if class_name is not None:
        sklearn_transformer = utils.load_obj(class_name)(**kwargs)
    wrapping_strategy = transformer_config.get("wrapper")
    if wrapping_strategy is None:
        return step_name, sklearn_transformer
    if wrapping_strategy == WrapperOptions.SELECT_COLUMNS.value:
        wrapped_transformer = SkLearnSelector(sklearn_transformer)
    elif wrapping_strategy == WrapperOptions.PRESERVE_COLUMNS.value:
        wrapped_transformer = SklearnTransform(sklearn_transformer)
    elif wrapping_strategy == WrapperOptions.PRESERVE_PANDAS.value:
        wrapped_transformer = ColumnNamesAsNumbers(sklearn_transformer)
    else:
        raise ValueError(
            f"Unknown value is set for wrapper: {wrapping_strategy}."
            " Should be either 'None' or between string options:"
            " 'select_columns', 'preserve_columns', 'preserve_pandas'",
        )
    return step_name, wrapped_transformer


def add_transformers(
    estimator: tp.Any,
    transformers: tp.Optional[tp.List[TTransformerStep]] = None,
    estimator_step_name: str = "estimator",
) -> sklearn.pipeline.Pipeline:
    """Creates a sklearn model pipeline based on the estimator and adds
    the desired transformer.

    The output Sklearn pipeline will the following order:
        1. Given transformers (e.g. feature selection/generation/scaling)
        2. Given estimator.

    Transformers can be specified by:
        - Config string accepted in ``load_transformer``.
        - Tuple of (pipeline step name, transformer object).

    Args:
        estimator: instantiated model object.
        transformers: optional list of transformer or transformer
            config dictionaries matching `TransformerInitConfig`
            to add to the SklearnPipeline.
        estimator_step_name: name of the estimator step for the sklearn.Pipeline.

    Returns:
        sklearn.pipeline.Pipeline with added transformers and estimator
    """
    transformer_steps = [
        load_transformer(transformer_config)
        for transformer_config in transformers or []
    ]
    return sklearn.pipeline.Pipeline(
        transformer_steps + [(estimator_step_name, estimator)],
    )
