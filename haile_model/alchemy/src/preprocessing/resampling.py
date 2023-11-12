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
from typing import Dict

import numpy as np
import pandas as pd

from optimus_core.tag_dict import TagDict

SUPPORTED_METHODS = ("mean", "min", "max", "first", "sum", "last", "median")

logger = logging.getLogger(__name__)


def resample_dataframe(
    data: pd.DataFrame,
    ts_col: str,
    resampling_method: Dict[str, str],
    **resample_kwargs: dict,
):
    """Resample dataframe columns according to resampling methods.

    Applies resampling_method to each column separately to
    resample to specified resample_freq.

    Args:
        data: input dataframe,
        ts_col: name of the timestamp column,
        resampling_method: dictionary mapping resampling function to columns
        resample_kwargs: keyword args to pass to pd.DataFrame.resample

    Returns:
        data_resampled: resampled output data
    """
    data[ts_col] = pd.to_datetime(data[ts_col])
    data = data.set_index(ts_col, drop=True)
    data_resampled = data.resample(**resample_kwargs).apply(resampling_method)
    logger.info(
        "Resampling data using predefined data "
        f"aggregation methods: \n{resampling_method}",
    )
    return data_resampled.reset_index()


def resample_data(
    data: pd.DataFrame,
    td: TagDict,
    timestamp_col: str,
    errors="coerce",
    default_method="mean",
    resample_kwargs: Dict = None,
) -> pd.DataFrame:
    """Resample data to resample frequency.

    Resample according to agg_method defined in data dictionary.

    Args:
        data: input data
        td: data dictionary
        timestamp_col: timestamp column name to use as index
        errors: 'raise' or 'coerce'. default = 'coerce'.
        default_method: method to use when agg_method is missing from td
        resample_kwargs: resample kwargs used for pandas.DataFrame.resample

    Returns:
        data_resampled: resampled output data
    """

    resampling_methods = {}
    data_cols = data.drop(timestamp_col, axis=1).columns.values
    data_cols = np.intersect1d(data_cols, td.select())

    for col in data_cols:
        method = get_valid_agg_method(col, td, errors, default_method)
        resampling_methods[col] = method

    return resample_dataframe(
        data,
        timestamp_col,
        resampling_methods,
        **resample_kwargs,
    )


def get_valid_agg_method(tag: str, td: TagDict, errors: str, default_method: str):
    """Select valid aggregation method for a tag.

    Selects the aggregation method for a tag from the
    data dictionary. If not defined, raise error or default
    to a default aggregation method.

    Args:
        tag: string of the tag
        td: data dictionary
        errors: str {'raise', 'coerce'}, raise errors if tag has no agg_method
            defined in data dictionary or coerce to default
        default_method: method to use when agg_method is missing from td

    Returns:
        data_resampled: resampled output data
    """

    method = td[tag]["agg_method"]

    if method in SUPPORTED_METHODS:
        return method

    if errors == "raise":
        if method is np.nan:
            raise ValueError(f"No aggregation method defined for column {tag}")
        raise ValueError(f"Invalid aggregation method defined for column {tag}")
    elif errors == "coerce":
        return default_method
    raise ValueError("Error behavior for missing agg_method not well defined")
