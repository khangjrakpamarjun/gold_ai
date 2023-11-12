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

from ..model_base import ProducesShapFeatureImportance

FEATURE_IMPORTANCE_COLUMN_NAME = "shap_feature_importance"
FEATURES_INDEX_NAME = "feature_name"


def calculate_shap_feature_importance(
    data: pd.DataFrame,
    shap_producer: ProducesShapFeatureImportance,
    **kwargs: tp.Any,
) -> pd.DataFrame:
    """Extract SHAP feature importances from instance of ``ProducesShapFeatureImportance``

    Args:
        data: data to be passed into
         ``ProducesShapFeatureImportance.get_shap_feature_importance``.
        shap_producer: an instance of ``ProducesShapFeatureImportance``
        kwargs: kwargs to be passed into ``get_shap_feature_importance``

    Returns:
        DataFrame of SHAP feature importance with feature names as the index.
    """
    shap_feature_importance = shap_producer.get_shap_feature_importance(data, **kwargs)
    return pd.DataFrame(
        data={FEATURE_IMPORTANCE_COLUMN_NAME: shap_feature_importance.values()},
        index=pd.Index(shap_feature_importance.keys(), name=FEATURES_INDEX_NAME),
    ).sort_values(by=FEATURE_IMPORTANCE_COLUMN_NAME, ascending=False)
