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
Preprocessing utils code
"""
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from optimus_core import TagDict


def create_summary_table(
    data: pd.DataFrame,
    td: Optional[TagDict] = None,
    percentile: float = 0.05,
) -> pd.DataFrame:
    """This function create descriptive summary table for input data

    Args:
        data (pd.DataFrame): input data.
        td (TagDict, optional): tag dictionary. Defaults to None.
        percentile (float, optional): percentile to use as threshold for oulier check.
             When tag dictionary is missing, (range_min, range_max)
             will be (percentile, 1-percentile). Defaults to 0.05.

    Returns:
        pd.pd.DataFrame: _description_
    """

    pd_desc = data.describe().T
    pd_desc["null_count"] = data.isnull().sum()
    pd_desc["inf_count"] = data[data == np.inf].count()

    outlier_count = count_outlier(data, td=td, percentile=percentile)

    resulting_df = pd_desc.join(outlier_count, how="left")

    resulting_df["%_below_range_min"] = (
        100 * resulting_df["below_range_min_count"] / resulting_df["count"]
    )

    resulting_df["%_above_range_max"] = (
        100 * resulting_df["above_range_max_count"] / resulting_df["count"]
    )

    return resulting_df


def create_range_map(td: TagDict) -> Dict[str, Tuple[float, float]]:
    """This function creates dictionary of range for each tag in tag dictionary.

    Args:
        td (TagDict): tag dictionary

    Returns:
        Dict: range map
    """

    range_map = {}

    for tag in td.select("tag"):
        range_map[tag] = (
            td[tag]["range_min"],
            td[tag]["range_max"],
        )

    return range_map


def count_outside_threshold(
    series: pd.Series,
    threshold: float,
    direction: str = "lower",
) -> float:
    """This function count number of outliers from given series.

    Args:
        series (pd.Series): series of data to check outliers
        threshold (float): threshold
        direction (str, optional): lower/upper direction
            to count outliers. Defaults to "lower".

    Raises:
        ValueError: raises when wrong direction is given.

    Returns:
        float: number of outliers
    """

    if direction == "lower":
        return np.sum(series < threshold)
    if direction == "upper":
        return np.sum(series > threshold)
    raise ValueError("direction must be either 'lower' or 'upper'")


def count_outlier(
    data: pd.DataFrame,
    td: Optional[TagDict] = None,
    percentile: float = 0.05,
) -> pd.DataFrame:
    """This function count outliers from given data

    Args:
        data (pd.DataFrame): input data
        td (TagDict, optional): tag dictionary. Defaults to None.
        percentile (float, optional): percentile to use as
            threshold for oulier check.
            when tag dictionary is missing, (range_min, range_max)
            will be (percentile, 1-percentile).
            Defaults to 0.05.

    Returns:
        pd.DataFrame: _description_
    """

    numeric_cols = data.select_dtypes("number").columns
    description_df = pd.DataFrame(index=numeric_cols)

    if td is not None:
        range_map = create_range_map(td)
    else:
        range_map = {}

    for tag in numeric_cols:

        (range_min, range_max) = range_map.get(
            tag,
            [
                data[tag].quantile(q=percentile),
                data[tag].quantile(q=(1 - percentile)),
            ],
        )

        description_df.loc[tag, "below_range_min_count"] = count_outside_threshold(
            data[tag],
            threshold=range_min,
            direction="lower",
        )

        description_df.loc[tag, "above_range_max_count"] = count_outside_threshold(
            data[tag],
            threshold=range_max,
            direction="upper",
        )

    return description_df
