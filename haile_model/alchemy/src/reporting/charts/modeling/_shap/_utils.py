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
from dataclasses import dataclass

import numpy as np

from reporting.api.types import ShapExplanation


@dataclass
class FeatureShapExplanation(object):
    """
    Stores feature's data and shap values

    Attributes:
        values: contains shap values
        data: contains initial feature data
    """

    # Using ``values`` to keep the same interface as shap.Explanations
    values: np.array  # noqa: WPS110
    data: np.array


def extract_explanations_for_given_features(
    features: tp.List[str],
    shap_explanation: ShapExplanation,
) -> tp.Dict[str, FeatureShapExplanation]:
    explanation_for_feature = {}
    for feature in features:
        feature_index_in_shap_data = shap_explanation.feature_names.index(feature)
        explanation_for_feature[feature] = FeatureShapExplanation(
            data=shap_explanation.data[:, feature_index_in_shap_data].astype(float),
            values=shap_explanation.values[:, feature_index_in_shap_data],
        )
    return explanation_for_feature


def sort_features(
    features: tp.List[str],
    order_by: np.ndarray,
    shap_explanation: ShapExplanation,
    descending: bool,
) -> tp.List[str]:
    if order_by is None:
        order_by = _get_order_by_mean_abs_shaps(features, shap_explanation)
    if len(features) != len(order_by):
        raise ValueError(
            "Please provide same length collections for `order_by` and `features`",
        )
    features = np.array(features)[np.argsort(order_by)].tolist()
    if descending:
        features.reverse()
    return features


def _get_order_by_mean_abs_shaps(
    features: tp.List[str],
    shap_explanation: ShapExplanation,
) -> np.ndarray:
    """
    Returns an array where the i-th value corresponds to the average absolute shap value
    for the i-th feature.
    """
    mean_shap_values = np.abs(shap_explanation.values).mean(axis=0)
    order_by = np.zeros_like(features, dtype=np.float64)
    for index, feature in enumerate(features):
        index_in_shaps = shap_explanation.feature_names.index(feature)
        order_by[index] = mean_shap_values[index_in_shaps]
    return order_by
