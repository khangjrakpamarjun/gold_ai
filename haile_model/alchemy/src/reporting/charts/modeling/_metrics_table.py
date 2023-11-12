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

from reporting.api.types import TFigure
from reporting.charts.primitive._table import plot_table


def plot_train_test_metrics(
    train_set_metrics: tp.Mapping[str, float],
    test_set_metrics: tp.Mapping[str, float],
    table_width: float,
) -> TFigure:
    train_and_test_metrics = pd.DataFrame.from_dict(
        {"train": train_set_metrics, "test": test_set_metrics},
        orient="index",
    )
    split_column = "Split"
    train_and_test_metrics.index.name = split_column
    return plot_table(
        data=train_and_test_metrics,
        width=table_width,
        sort_by=[(split_column, "desc")],
        columns_filters_position=None,
    )
