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
Timezone resolution code
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def round_timestamps(
    frequency: str,
    data: pd.DataFrame,
    datetime_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Rounds timestamps in order to reduce minor timestamp noise.
    Different frequency aliases can be found here:
    https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases

    Args:
       frequency: timeseries offset aliases.
       data: input data
       datetime_col: timestamp column

    Returns:
       data with rounded timestamps
    """
    data = data.copy()
    data[datetime_col] = pd.to_datetime(data[datetime_col]).dt.round(frequency)
    logger.info(f"Rounding '{datetime_col}' to '{frequency}' frequency.")
    return data
