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
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from optimus_core.tag_dict import TagDict
from optimus_core.utils import load_obj

logger = logging.getLogger(__name__)
quality_check_logger = logging.getLogger("quality_check_logger")

TSeriesModifier = Callable[[pd.Series], pd.Series]


def replace_inf_values(data: pd.DataFrame) -> pd.DataFrame:
    """Replace any infinite values in dataset with NaN.

    Args:
        data: input data

    Returns:
        Dataframe with infinite values replaced by NaN & dropped only if explicitly
        asked to drop those
    """
    df_new = data.copy()
    infinity_set = [np.inf, -np.inf]
    df_new = df_new.replace(infinity_set, np.nan)
    summary = pd.DataFrame()
    summary["before_cleaning"] = data.isin(infinity_set).sum()
    summary["after_cleaning"] = df_new.isin(infinity_set).sum()

    summary_count = summary.loc[summary["before_cleaning"] > 0]

    logger.info(f"\nnumber of inf values in data: \n{summary_count}")

    return df_new


def deduplicate_pandas(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Drop duplicates for pandas dataframe

    Args:
       data: input data
       **kwargs: keywords feeding into the pandas `drop_duplicates`

    Returns:
       data with duplicates removed
    """
    logger.info(f"Dataframe shape before dedup: {data.shape}")

    sub = data.drop_duplicates(**kwargs)
    sub.reset_index(inplace=True, drop=True)

    logger.info(f"Dataframe shape after dedup: {sub.shape}")

    n_dropped = data.shape[0] - sub.shape[0]
    if n_dropped > 0:
        quality_check_logger.info(f"Dropped {n_dropped} duplicate timestamps")
    else:
        quality_check_logger.info("No duplicate timestamps in data source.")
    return sub


def unify_timestamp_col_name(
    data: pd.DataFrame,
    datetime_col: str,
    unified_name: str = "timestamp",
) -> pd.DataFrame:
    """Unify all timestamp column names that will be further used as index

    Args:
       params: dictionary of parameters
       data: input data
       unified_name: desired unified column name

    Returns:
       data
    """

    # check if a duplicate unified_name will be created
    # raise an error if so
    if (unified_name in data.columns) and (unified_name != datetime_col):
        raise ValueError(
            f"column name '{unified_name}' already exists. "
            f"Renaming another column to '{unified_name}' "
            f"will lead to duplicate column names",
        )

    df = data.rename(columns={datetime_col: unified_name})
    logger.info(f"Rename column '{datetime_col}' to '{unified_name}'.")

    return df


def remove_null_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Identify columns that contain all NaN/NaTs and drop them

    Args:
       data: input pandas dataframe

    Returns:
       data
    """
    # Get initial columns from data
    data_columns = set(data.columns)

    # Drop all columns that are comprised of ALL NaN's/NaT's
    data = data.dropna(axis=1, how="all")

    # Get a set of all the columns that have been dropped
    dropped_columns = data_columns.difference(set(data.columns))

    if dropped_columns:
        logger.info(
            f"Dropped columns: {dropped_columns} due to all"
            " column values being NaN/NaT",
        )
        quality_check_logger.info(
            f"Dropped columns: {dropped_columns} due to all"
            " column values being NaN/NaT",
        )
    else:
        logger.info("All columns have values. Continuing...")
    return data


def enforce_custom_schema(
    data: pd.DataFrame,
    data_types: Dict[str, Union[str, TSeriesModifier]],
    map_str_bool: Dict[str, str] = None,
    schema_func: Dict[str, Dict[str, Callable]] = None,
) -> pd.DataFrame:
    """Apply schema to certain columns for pandas dataframe

    Args:
       data: input data
       data_types: column names and their corresponding data types
       map_str_bool: optional argument for mapping strings to boolean values
       schema_func: Optional additional schema conversions functions.

    Returns:
       df
    """
    df = data.copy()
    schema_grp = defaultdict(list)
    for column, data_type in sorted(data_types.items()):
        schema_grp[data_type].append(column)

    # datetime conversion is taken care of in the timezone conversion
    # so datetime columns are excluded here
    schema_grp.pop("datetime", None)
    # apply corresponding types
    for type_choice in schema_grp.keys():
        apply_type(df, schema_grp, type_choice, map_str_bool, schema_func)

    return df


def apply_type(
    df: pd.DataFrame,
    schema_grp: Dict[str, List[str]],
    type_choice: str,
    map_str_bool: Dict[str, str],
    schema_func: Dict[str, Dict[str, Callable]],
) -> None:
    """Apply schema to certain column

    Args:
        df: input data
        schema_grp: data types and their corresponding columns
        type_choice: dtypes available
        map_str_bool: optional dictionary of mapping for str to boolean values
        schema_func: optional dictionary containing
            path to any custom conversion function

    Returns:
       None
    """
    current_type_cols = schema_grp.get(type_choice)

    if not isinstance(current_type_cols, list):
        return

    conversions = {
        "numeric": {
            "func": lambda x: pd.to_numeric(x, errors="coerce"),  # noqa: WPS111
        },
        "categorical": {
            "func": lambda x: x.astype("category"),  # noqa: WPS111
        },
        "boolean": {
            "func": lambda x: series_convert_bool(x, map_str_bool),  # noqa: WPS111
        },
    }
    conversions = conversions.update(schema_func) if schema_func else conversions

    logger.info(
        f"Converting {current_type_cols} columns" f" to '{type_choice}' data type.",
    )

    df[current_type_cols] = df[current_type_cols].apply(**conversions[type_choice])


def series_convert_bool(col, map_str_bool: Dict[str, str] = None):
    if map_str_bool:
        return col.str.lower().map(map_str_bool)
    return col.apply(convert_bool)


def convert_bool(prompted_value: str) -> float:
    """convert different boolean candidates to 0 or 1

    Args:
        prompted_value: input value

    Raises:
        ValueError: if the string value does not correspond to a boolean.

    Returns:
       0 or 1
    """
    prompted_value = str(prompted_value).lower()
    if prompted_value in {"yes", "true", "t", "1", "on", "1.0"}:
        return 1
    if prompted_value in {"no", "false", "f", "0", "off", "0.0"}:
        return 0
    if prompted_value == "nan":
        return np.NaN
    raise ValueError(f"Invalid boolean value {prompted_value}")


def enforce_schema(
    data: pd.DataFrame,
    td: TagDict,
    map_str_bool: Dict = None,
    schema_func_param: Dict[str, Dict[str, Callable]] = None,
) -> pd.DataFrame:
    """Enforce schema based on data types defined in tagdict, including
    "numeric", "categorical", "boolean". "datetime" columns are
    handled in the node `set_timezones`

    Args:
        data: input data
        td: tag dictionary
        map_str_bool: optional dictionary of mapping for str to boolean values
        schema_func_param: optional dictionary containing path
            to any custom conversion function

    Returns:
       data
    """
    data_types = {}
    for col in data.columns:
        if col in td:
            data_types[col] = td[col]["data_type"]  # noqa: WPS529

    if schema_func_param:
        schema_func = {
            schema_func_param.get("data_type"): {
                "func": load_obj(schema_func_param.get("func")),
            },
        }
        data = enforce_custom_schema(data, data_types, map_str_bool, schema_func)
    else:
        data = enforce_custom_schema(data, data_types, map_str_bool, None)

    return data


def remove_outlier(
    data: pd.DataFrame,
    td: TagDict,
    rule: str = "clip",
) -> pd.DataFrame:
    """Remove outliers based on value range set in tagdict
    and selected rule

    Args:
       data: input data
       td: tag dictionary
       rule: ways to remove outlier. either 'clip' or 'drop'

    Returns:
       df_new
    """
    # construct a dict with col name and its value range
    tag_range = {}
    for col in data.columns:
        if col in td:
            tag_range[col] = (
                td[col]["range_min"],  # noqa: WPS529
                td[col]["range_max"],  # noqa: WPS529
            )

    # apply outlier removing rule
    df_new = apply_outlier_remove_rule(data.copy(), rule, tag_range)

    if rule == "drop":
        n_dropped = df_new.isnull().sum() - data.isnull().sum()
        perc_dropped = n_dropped / data.shape[0] * 100

        summary = pd.DataFrame()
        summary["n_dropped"] = n_dropped
        summary["perc_dropped(%)"] = perc_dropped
        dropped_outliers = summary.loc[summary["n_dropped"] > 0].round(2)
        quality_check_logger.info(
            f"number of outliers dropped per column: \n{dropped_outliers}",
        )

    return df_new


def apply_outlier_remove_rule(  # noqa: WPS231
    df: pd.DataFrame,
    rule: str,
    num_tag_range: Dict[str, Tuple[float, float]],
) -> pd.DataFrame:
    """Remove outliers with selected rule

    Args:
       df: input data
       rule: ways to remove outlier. either 'clip' or 'drop'
       num_tag_range: dict with col name and its value range

    Returns:
       df
    """
    for col in df.columns:
        # skip columns that are not numeric
        if col not in num_tag_range.keys():
            continue

        td_low, td_up = num_tag_range[col]

        # skip columns that don't have lower and upper limits
        if np.isnan(td_low) and np.isnan(td_up):
            continue

        lower = None if np.isnan(td_low) else td_low
        upper = None if np.isnan(td_up) else td_up

        if rule == "clip":
            df[col].clip(lower, upper, inplace=True)
            logger.info(f"Clipping {col} to [{lower}, {upper}] range.")
        elif rule == "drop":
            outside_range_mask = (df[col] < lower) | (df[col] > upper)  # noqa: WPS465
            df[col].mask(outside_range_mask, inplace=True)
        else:
            raise ValueError(
                f"Invalid outlier removal rule `{rule}`. "
                "Choose supported rules 'clip' or 'drop'",
            )

    return df
