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


import logging

import pandas as pd

logger = logging.getLogger(__name__)


def add_time_step_column(
    sensor_data: pd.DataFrame,
    timestamp: str,
    batch_id_col: str,
    time_step_column: str,
) -> pd.DataFrame:
    """
    Creates `time_step_column` as a range starting from 0 = min(datetime)

    Args:
        sensor_data: sensors' time series; each timestamp is assigned to a batch
        timestamp: timestamp column name
        batch_id_col: batch_id column name
        time_step_column:
    """

    sensor_data = (
        sensor_data.sort_values([batch_id_col, timestamp]).reset_index(drop=True).copy()
    )

    if time_step_column in sensor_data.columns:
        raise ValueError(
            f"Provided `time_step_column='{time_step_column}'` already exists "
            f"and will be overwritten.",
        )

    sensor_data[time_step_column] = sensor_data.groupby(batch_id_col).cumcount()
    return sensor_data
