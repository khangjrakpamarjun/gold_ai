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

import numpy as np
import pandas as pd
import shap

from ._datasets import CustomModelWrapper  # noqa: WPS436


def get_shap_explanation(
    model: CustomModelWrapper,
    data: pd.DataFrame,
) -> shap.Explanation:
    explanation = _explain_using_generic_explainer(model=model, data=data)
    shap_values_for_original_columns = _extract_shap_values_for_original_columns(
        explanation=explanation,
        original_columns=data.columns,
    )
    return shap.Explanation(
        data=data.values,
        values=shap_values_for_original_columns.values,
        feature_names=data.columns,
        base_values=explanation.base_values,
    )


def _explain_using_generic_explainer(
    model: CustomModelWrapper,
    data: pd.DataFrame,
) -> shap.Explanation:
    data = data[model.features_in]
    explainer = shap.Explainer(model=model.estimator, masker=data)
    return explainer(data)


def _extract_shap_values_for_original_columns(
    explanation: shap.Explanation,
    original_columns: tp.List[str],
) -> pd.DataFrame:
    shape_for_explanation = (len(explanation.data), len(original_columns))
    shap_values = pd.DataFrame(
        data=np.zeros(shape_for_explanation, dtype=np.float64),
        columns=original_columns,
    )
    for feature_index, feature_name in enumerate(explanation.feature_names):
        shap_values[feature_name] = explanation.values[:, feature_index]
    return shap_values
