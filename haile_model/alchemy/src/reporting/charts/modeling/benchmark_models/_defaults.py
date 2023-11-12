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
A module with the default models used in the default reports.
"""
import typing as tp

import pandas as pd

from ._autoregressive import AutoregressiveBenchmarkModel
from ._benchmark_model_base import BenchmarkModelBase
from ._moving_average import MovingAverageBenchmarkModel


def fit_default_benchmark_models(
    target: str,
    data: pd.DataFrame,
    timestamp: str,
) -> tp.Dict[str, BenchmarkModelBase]:
    """Fits the default benchmark models."""
    benchmark_models = {
        "Autoregressive Model (AR1)": AutoregressiveBenchmarkModel(
            target=target,
            timestamp=timestamp,
        ).fit(data),
        "Moving Average Model (30D)": MovingAverageBenchmarkModel(
            target=target,
            timestamp=timestamp,
        ).fit(data),
    }
    return benchmark_models  # noqa: WPS331  # Naming makes meaning clearer
