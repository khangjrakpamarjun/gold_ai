import logging
from datetime import datetime, timedelta
from typing import Dict

import pandas as pd

from alchemy.pipelines.data_science.model_input.model_input_nodes import aggregate_data
from optimus_core import TagDict

logger = logging.getLogger(__name__)


def _cut_range_important_features(df: pd.DataFrame, cut_range: Dict):
    for feature, range_dict in cut_range.items():
        min_limit = range_dict.get("min", False)
        max_limit = range_dict.get("max", False)
        if min_limit:
            df = df.loc[df[feature] >= min_limit].copy()
        if max_limit:
            df = df.loc[df[feature] <= max_limit].copy()
    return df


def prepare_test_data(df: pd.DataFrame, td: TagDict, params: Dict):
    # Filter master data for number of days from last date based on testing parameters
    filter_date = str(
        pd.to_datetime(df.index.max().date() - timedelta(days=params["days_of_data"]))
    )
    df = df[df.index >= filter_date]

    # Aggregate data based on circuit
    df = aggregate_data(df, td, params["aggregation"])

    # Cut rows for variables outside of range
    cut_range = params.get("cut_range", {})
    df = _cut_range_important_features(df, cut_range)

    # Drop rows that have null target columns
    df = df.dropna(subset=params["target_columns"], axis=0)

    # Fetch only few records to test both pipeline
    df = df.sample(n=params["number_of_obs"])

    df.reset_index(inplace=True)

    return df
