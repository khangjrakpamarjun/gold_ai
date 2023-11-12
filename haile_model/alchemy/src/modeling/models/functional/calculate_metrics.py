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

from ..model_base import EvaluatesMetrics

METRICS_VALUE_COLUMN = "metric_value"
METRIC_NAME_COLUMN = "metric_name"


def calculate_metrics(
    data: pd.DataFrame,
    model: EvaluatesMetrics,
    **predict_kwargs: tp.Any,
) -> pd.DataFrame:
    """Create a DataFrame of common regression metrics.

    Predictions from ``prediction_column`` will be used. If ``prediction_column`` is
    not specified model will be used to generate predictions.

    Args:
        data: data to make predictions from
         or get predictions from ``prediction_column``.
        model: trained instance of ModelBase.
         Used to make predictions and indicate model target column
        predict_kwargs: keyword arguments to ``.predict`` method.

    Returns:
        DataFrame of with metric names as the index.
    """
    metrics = model.evaluate_metrics(data, **predict_kwargs)
    return pd.DataFrame(
        data={METRICS_VALUE_COLUMN: metrics.values()},
        index=pd.Index(metrics.keys(), name=METRIC_NAME_COLUMN),
    )
