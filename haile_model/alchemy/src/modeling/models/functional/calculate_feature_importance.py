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

from ..model_base import ModelBase

FEATURE_IMPORTANCE_COLUMN_NAME = "feature_importance"
FEATURES_NAME_COLUMN = "feature_name"


def calculate_feature_importance(
    data: pd.DataFrame,
    model: ModelBase,
    **kwargs: tp.Any,
) -> pd.DataFrame:
    """Extract feature importances from instance of ``ModelBase``

    Args:
        data: data to be passed into ``ModelBase.calculate_feature_importance``.
        model: a trained instance of ``ModelBase``
         in ``pd.DataFrame`` based on it's importance
        kwargs: kwargs to be passed into model's ``get_feature_importance_method``

    Returns:
        DataFrame of feature importance with feature names as the index.
    """
    feature_importance = model.get_feature_importance(data, **kwargs)
    return pd.DataFrame(
        data={FEATURE_IMPORTANCE_COLUMN_NAME: feature_importance.values()},
        index=pd.Index(feature_importance.keys(), name=FEATURES_NAME_COLUMN),
    ).sort_values(by=FEATURE_IMPORTANCE_COLUMN_NAME, ascending=False)
