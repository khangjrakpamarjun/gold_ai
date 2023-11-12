##########################################################################################
#                       Model input nodes
##########################################################################################

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from optimus_core import TagDict

logger = logging.getLogger(__name__)


def aggregate_data(
    df: pd.DataFrame, td: TagDict, aggregation_params: Dict
) -> pd.DataFrame:
    """
    Aggregate data using resampling
    Args:
        df: data to aggregate
        td: Tag dictionary to fetch aggregation method information
        aggregation_params: parameters used for aggregation
    Returns:
        df_agg: Aggregated data
    """
    # Extracting aggregation parameters (resampling parameters)
    resampling_freq = aggregation_params["resampling_freq"]
    offset = aggregation_params["offset"]
    closed = aggregation_params["closed"]
    label = aggregation_params["label"]
    notnull_cutoff = aggregation_params["notnull_minima"]

    # Obtaining tag to aggregation method mapping # TODO: reinstate this
    # aggregation_method_mapping = _process_tag_dict_for_aggregation(
    #     td, aggregation_params
    # )

    df_agg = df.resample(
        resampling_freq, offset=offset, label=label, closed=closed
    ).agg("mean")

    for feature in df.columns:
        # Verify min count for each aggregation period
        dummy_df = (
            df[feature]
            .resample(resampling_freq, offset=offset, label=label, closed=closed)
            .agg(["count", "size"])
        )
        dummy_df["perc_null"] = dummy_df["count"] / dummy_df["size"]

        df_agg.loc[dummy_df["perc_null"] < notnull_cutoff, feature] = np.nan
    return df_agg


def _process_tag_dict_for_aggregation(
    td: pd.DataFrame, aggregation_params: Dict
) -> Dict:
    """
    Function for extracting information about aggregation strategy from tag dictionary
    Args:
        td: tag dictionary
        aggregation_params: parameters used for aggregation
    Returns:
        aggregation_method_mapping: dictionary having tag to aggregation method mapping (eg - 'mean', 'last')
    """

    # Column name of aggregation method in tag dictionary
    aggregation_method = aggregation_params["aggregation_method"]

    return td[aggregation_method].dropna().to_dict()


def filter_data_by_timestamp(
    data: pd.DataFrame,
    params: Dict,
) -> pd.DataFrame:
    """
    Filter starting point of data

    Args:
        data (pd.DataFrame): incoming data
        params (Dict): [] for the starting date

    Returns:
        pd.DataFrame: merged dataset
    """
    time_period_params = params["time_period"]

    df = data.copy()

    if time_period_params["period_method"] == "date":
        if (time_period_params["start_date"] != "") & (
            time_period_params["end_date"] != ""
        ):
            df = df.loc[df.index >= time_period_params["start_date"]]
            df = df.loc[df.index <= time_period_params["end_date"]]
        elif (time_period_params["start_date"] != "") & (
            time_period_params["end_date"] == ""
        ):
            df = df.loc[df.index == time_period_params["start_date"]]
        else:
            msg = f"Provide right start and end date"
            logger.error(msg)
            raise ValueError(msg)
    elif time_period_params["period_method"] == "n_end_shifts":
        if time_period_params["end_date"] != "":
            df = df.loc[df.index <= time_period_params["end_date"]]
        n_end_shifts = time_period_params["n_end_shifts"]
        df = df.iloc[-n_end_shifts:, :]
    else:
        msg = "Unexpected value for 'period_method' parameter"
        logger.error(msg)
        raise ValueError(msg)

    # if params.get("remove_other_periods"):  # TODO: check if needed
    #     remove_other_periods = params.get("remove_other_periods")
    #     if remove_other_periods["remove"]:
    #         start_date = remove_other_periods["start_date_filter"]
    #         end_date = remove_other_periods["end_date_filter"]
    #         mask = (df[params["datetime_col"]] > start_date) & (
    #             df[params["datetime_col"]] < end_date
    #         )
    #         df = df[~mask]
    #         logger.info(f"\n\nShape after remove_other_periods {df.shape}")

    # Check dataset not empty
    if len(df) == 0:
        msg = "Dataset is empty after applying date filters"
        logger.error(msg)
        raise ValueError(msg)

    return df


def filter_data_by_target(
    params: Dict, td: pd.DataFrame, data: pd.DataFrame
) -> pd.DataFrame:
    """
    Filter out rows based on features and their allowed range defined in params.
    Args:
        params: Model input parameters.
        td: Tag dictionary.
        data: Data

    Returns: Data after removing rows.
    """
    # TODO: PS to remove hard coded value
    if params["target_column"] == "recovery" and pd.isna(data["recovery"][-1]):
        data["recovery"][-1] = 0.80
    # Remove NaN in target column
    df = data.dropna(subset=params["target_column"], axis=0)
    n_before = df.shape[0]
    target_column = params["target_column"]
    if n_before < 1:
        msg = (
            f"Target column feature {target_column} contains NA value in the last shift"
            f" of the data. Please check data quality for {target_column} feature "
        )
        logger.error(msg)
        raise ValueError(msg)

    # Cut rows for variables outside of range
    cut_range = params.get("cut_range", {})
    if n_before == 1:  # for live recs
        for feature, range_dict in cut_range.items():
            min_limit = range_dict.get("min", False)
            max_limit = range_dict.get("max", False)

            if np.isnan(df[feature][0]):
                msg = (
                    f"Feature {feature} has value {df[feature][0]} in the last shift."
                    " Last shift data removed because of nan value in feature"
                    f" {feature}. Please check data quality for {feature} feature."
                )
                logger.error(msg)
                raise ValueError(msg)
            if min_limit:
                df_filtered = df.loc[df[feature] >= min_limit].copy()
                if df_filtered.shape[0] < 1:
                    msg = (
                        f"Feature {feature} has value {df[feature][0]} which is less"
                        f" than min_limit {min_limit} in cut_range. Last shift data"
                        " removed because of this . Please check data quality for"
                        f" {feature} feature."
                    )
                    logger.error(msg)
                    raise ValueError(msg)
            if max_limit:
                df_filtered = df.loc[df[feature] <= max_limit].copy()
                if df_filtered.shape[0] < 1:
                    msg = (
                        f"Feature {feature} has value  {df[feature][0]}  which is"
                        f" greater than max_limit {max_limit} in cut_range. Last shift"
                        " data removed because of this. Please check data quality for"
                        f" {feature} feature."
                    )
                    logger.error(msg)
                    raise ValueError(msg)

    else:  # for cfa
        for feature, range_dict in cut_range.items():
            min_limit = range_dict.get("min", False)
            max_limit = range_dict.get("max", False)
            if min_limit:
                df = df.loc[df[feature] >= min_limit].copy()
                n_after = df.shape[0]
                # Check that at least 60% of data persisted after removal
                if n_after / n_before < 0.6:
                    msg = (
                        f"Filter removed {100 * (1 - n_after / n_before)}% of the data"
                        f" as feature {feature} has value less than min_limit in"
                        " cut_range."
                    )
                    logger.error(msg)
                    raise ValueError(msg)

            if max_limit:
                df = df.loc[df[feature] <= max_limit].copy()
                n_after = df.shape[0]
                # Check that at least 60% of data persisted after removal
                if n_after / n_before < 0.6:
                    msg = (
                        f"Filter removed {100 * (1 - n_after / n_before)}% of the data"
                        f" as feature {feature} has value greater than max_limit in"
                        " cut_range."
                    )
                    logger.error(msg)
                    raise ValueError(msg)
    n_after = df.shape[0]
    # Check that overall at least 60% of data persisted after removal
    if n_after / n_before < 0.6:
        msg = f"Filter removed {100 * (1 - n_after / n_before)}% of the data"
        logger.error(msg)
        raise ValueError(msg)
    df.index.name = params["datetime_column"]
    df.reset_index(inplace=True)

    return df
