import logging
from typing import Dict, List

import pandas as pd
import pytz

from preprocessing import deduplicate_pandas, round_timestamps

logger = logging.getLogger(__name__)


def _convert_timezone_to_utc(df, ts_col, timezone):
    """
    Converts data to required timezone
    Args:
        df: data
        ts_col: timestamp column name
        timezone: required timezone value

    Returns: data with required timezone
    """
    tz = pytz.timezone(timezone)
    df.set_index(ts_col, inplace=True)

    df.index = pd.to_datetime(df.index).tz_localize(tz)
    return df


def _format_timestamp(df, ts_col):
    """
    Formats the timestamp column of the dataframe
    Args:
        df: data
        ts_col: timestamp column name

    Returns: data with formatted timestamp
    """
    df = df.reset_index()
    df[ts_col] = pd.to_datetime(df[ts_col], format="%Y-%m-%d %H:%M:%S")
    df.sort_values(by=[ts_col], inplace=True)
    return df


def _unify_timestamp_column_name(df, ts_col, datetime_col):
    """
    Replaces the timestamp column value with the required name
    Args:
        df: data
        ts_col: timestamp column name
        datetime_col: column name which needs to be replaced

    Returns: data with required timestamp column name
    """
    if ts_col == datetime_col:
        return df
    return df.rename(columns={datetime_col: ts_col})


def _standardize_time_index(
    df: pd.DataFrame, start: str, end: str, freq: str, name: str
) -> pd.DataFrame:
    """
    Standardizes time index leaving no gaps between start and end timestamps.
    Args:
        df: data
        start: start timestamp
        end: end timestamp
        freq: frequency in index
        name: name of index

    Returns: standardized DataFrame

    """
    if df.index.duplicated().sum() > 0:
        raise ValueError("Duplicates found in df")
    dates_range = pd.date_range(start=start, end=end, freq=freq)
    output_df = pd.DataFrame(index=dates_range)
    output_df = output_df.join(df, how="left")
    output_df.index.names = [name]
    return output_df


def _ingestion_cleaning(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Function to perform data cleaning during ingestion phase.
    It cleans the timestamp column
    Args:
        df: Raw data
        params: Raw file parameters for timestamp

    Returns: Processed data with cleaned timestamp

    """
    logger.info("Initialising Ingestion Cleaning Process")
    ts_col = params["timestamp_column"]

    logger.info("Aligning Timezone to UTC")
    df = _convert_timezone_to_utc(df, ts_col, params["timezone_value"])

    logger.info("Formatting the timestamp of the data")
    df = _format_timestamp(df, ts_col)

    logger.info("Rounding of timestamp as per the provided frequency")
    df = round_timestamps(params["pipeline_frequency"], df, ts_col)

    logger.info("Renaming the timestamp column")
    df = _unify_timestamp_column_name(df, params["timestamp_column"], ts_col)

    logger.info("Removing duplicates from the dataset")
    df = deduplicate_pandas(df)

    df.set_index(ts_col, inplace=True)
    logger.info("Standardizing the time index")
    df = _standardize_time_index(
        df=df,
        start=params["data_start_time"],
        end=df.index.max(),
        freq=params["pipeline_frequency"],
        name=ts_col,
    )

    logger.info("Ingestion Cleaning Process completed")
    return df


def ingest_pi_data(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Function to process pi_data and sort the data on the basis of timestamp column
    Args:
        df: Raw pi data
        params: Raw file parameters for timestamp

    Returns: Processed pi data with formatted timestamp

    """
    return _ingestion_cleaning(df, params)


def ingest_raw_pi_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to ingest timescale data in raw layer
    Args:
        df: Raw pi data

    Returns: Processed pi data

    """
    return df


def find_active_features(td: pd.DataFrame) -> List:
    """Returns a list of target and model features for all models in td"""

    is_active = td.filter(regex="feature|target").sum(axis=1)
    return is_active.loc[is_active > 0].index.to_list()
