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

from __future__ import annotations

import logging
import typing as tp
from collections import abc

import pandas as pd

logger = logging.getLogger(__name__)


def check_columns_exist(
    data: pd.DataFrame,
    col: tp.Union[str, tp.Iterable[str]] = None,
) -> None:
    if isinstance(col, str):
        if col not in data.columns:
            raise ValueError(f"Column {col} is not included in the dataframe.")

    if isinstance(col, abc.Iterable):
        columns = set(col)
        if not columns.issubset(data.columns):
            not_included_cols = columns - set(data.columns)
            raise ValueError(
                "The following columns are missing"
                f" from the dataframe: {not_included_cols}.",
            )
