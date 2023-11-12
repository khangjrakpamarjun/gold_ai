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

MODEL_PREDICTION = "model_prediction"


def calculate_model_predictions(
    data: pd.DataFrame,
    model: ModelBase,
    **predict_kwargs: tp.Any,
) -> pd.DataFrame:
    """Append predictions for the given model to the provided static_features.

    Args:
        data: dataset for making predictions with model
        model: trained instance of ``ModelBase``
        predict_kwargs: keyword arguments to predict function.

    Returns:
        Copy of DataFrame with an additional column for predictions.
    """
    predictions = model.predict(data, **predict_kwargs)
    return pd.DataFrame(
        data={MODEL_PREDICTION: predictions},
        index=data.index,
    )
