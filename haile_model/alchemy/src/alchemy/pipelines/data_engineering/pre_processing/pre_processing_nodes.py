import logging
import re
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from alchemy.pipelines.data_engineering.ingestion.ingestion_nodes import (
    find_active_features,
)
from optimus_core import TagDict
from preprocessing import enforce_schema, remove_null_columns, replace_inf_values

logger = logging.getLogger(__name__)


################################################################################
#                           Merge ingested datasets
################################################################################
def merge_datasets(**dict_of_df: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge ingested pre-processed data. If dict_of_df has one dataframe
    then return that dataframe as is otherwise merge the all the datasets
    inside dict_of_df dynamically into a single table.
    Args:
        dict_of_df: Dictionary of dataframes
    Returns: Returns merged dataframe

    """
    list_of_dataframes = [pd.DataFrame(dict_of_df[x]) for x in dict_of_df.keys()]

    if len(list_of_dataframes) == 1:
        return list_of_dataframes[0]

    return reduce(
        lambda df1, df2: pd.merge(
            df1, df2, left_index=True, right_index=True, how="outer", validate="1:1"
        ),
        list_of_dataframes,
    )


################################################################################
#                           Clean data columns
################################################################################
def clean_data(df: pd.DataFrame, td, outlier_params) -> pd.DataFrame:
    """
    Function to clean the merged dataset.
    Args:
        df: Merged Dataframe
        params: Clean Column parameters
        td: tag dictionary
        outlier_params: parameters for handling outliers
    Returns: Cleaned merged dataframe

    """

    df = df.copy()
    logger.info("Initialising Pre Processing CLeaning Process")

    # Replace any infinity values with NaN
    logger.info("Replacing Infinity Values with NaN")
    df = replace_inf_values(df)

    # TODO: Add Remove Negative Values Functionality
    # df = _remove_negative_values(df)

    logger.info("Checking Null Columns")
    df = remove_null_columns(df)

    # TODO: Add Clean Column Functionality
    # Convert all columns to lower case. Clean column names ->
    # snake case converts all columns into small case
    # separated by underscores and removes any special characters
    # replace_strings_params = params["replace_strings"]
    # df.columns = df.columns.str.lower()
    # df = clean_columns(df, case="snake", replace=replace_strings_params)

    logger.info("Enforcing Schema as per Tag Dictionary")
    df = enforce_schema(df, td)
    # TODO: Add Set Equipment On Off Functionality
    # TODO: Include Remove Shutdown Period Functionality

    return df


def _process_tag_dict_for_removing_outliers(
    td: pd.DataFrame, outlier_params: Dict
) -> Dict:
    """
    Function for extracting information about data source and outlier
    strategy from tag dictionary
    Args:
        td: tag dictionary
        outlier_params: parameters.remove_outliers used for outlier removal
    Returns:
        outlier_method_mapping: dictionary having tag to outlier strategy mapping
        data_source_mapping: dictionary having tag to data source mapping
    """
    # Column name of outlier strategy in tag dictionary
    outlier_strategy = outlier_params["outlier_strategy"]

    # Drop rows of tag dictionary where outlier strategy or tag id are null
    td1 = td[
        (td["range_min"].notna() & td["range_max"].notna())
        | td[outlier_strategy].notna()
    ]

    # Extracting outlier treatment methods from tag dictionary
    outlier_method_mapping = td1[["range_min", "range_max", outlier_strategy]].to_dict(
        orient="index"
    )

    return outlier_method_mapping


def remove_outliers(
    data: pd.DataFrame, td: TagDict, outlier_params: Dict
) -> Dict[str, pd.DataFrame]:
    """
    Remove outliers based on strategy defined in tag dictionary.

    If range min/max are specified in dict, this method takes precedence.
    Else, the outlier strategy is derived from the 'outlier_cleaning_strategy'
    column and inter-quartile range method is used.
    Otherwise, no outlier removal is performed.

    Args:
        data: Merged feature/input dataframe
        outlier_params: remove outliers parameters
        td: tag dictionary
    Returns:
        treated_data: Outlier cleaned merged feature/input dataframe (30min)
        min_max_dict: Report of min/max ranges and outlier strategy
    """
    logger.info("Removing Outliers from the dataset")

    td = td.to_frame()

    df = data

    outlier_method_mapping = _process_tag_dict_for_removing_outliers(
        td[td["derived"] == outlier_params["derived_column"]].set_index("tag"),
        outlier_params,
    )

    # Initialize empty min_max_dict for each feature
    min_max_dict = {
        feat: {"range_min": np.nan, "range_max": np.nan}
        for feat in outlier_method_mapping.keys()
    }

    # outlier_cleaning_threshold = outlier_params["max_percent"]

    # Build min_max_dict
    for feature in data.columns:
        # Iterate over columns
        if feature in outlier_method_mapping:
            # Check cleaning method for each feature
            if not np.isnan(outlier_method_mapping[feature]["range_min"]):
                # If range_min exists, it is assumed both min/max exist
                min_max_dict[feature]["range_min"] = outlier_method_mapping[feature][
                    "range_min"
                ]
                min_max_dict[feature]["range_max"] = outlier_method_mapping[feature][
                    "range_max"
                ]
            elif not pd.isnull(
                outlier_method_mapping[feature]["outlier_cleaning_strategy"]
            ):
                # Otherwise, use quantiles
                strategy = outlier_method_mapping[feature]["outlier_cleaning_strategy"]
                p1 = int(strategy.split("_")[0][0:2])  # Lower percentile
                p2 = int(strategy.split("_")[1][0:2])  # Upper percentile
                q_1, q_3 = np.nanpercentile(data[feature], [p1, p2])
                iqr = q_3 - q_1
                lower_bound = q_1 - (iqr * 1.5)
                upper_bound = q_3 + (iqr * 1.5)
                min_max_dict[feature]["range_min"] = lower_bound
                min_max_dict[feature]["range_max"] = upper_bound
            else:
                # If no cleaning specified, skip feature
                continue
    # Clean dataset based on min_max_dict
    treated_data = _clean_data_from_min_max_dict(data=data, min_max_dict=min_max_dict)
    # Check whether too much data was removed from a variable
    # TODO: need to fix issues with min/max ranges in td
    # _validate_pct_data_removed(
    #     data_start=data, data_end=treated_data,
    #     thresh=outlier_cleaning_threshold, td=td
    # )

    # TODO: this function should also create a full report of % values cleaned

    logger.info("Pre Processing Cleaning Process completed")
    return dict(
        df=treated_data,
        min_max_report=pd.DataFrame().from_dict(min_max_dict, orient="index"),
    )


def _validate_pct_data_removed(
    data_start: pd.DataFrame, data_end: pd.DataFrame, thresh: float, td: pd.DataFrame
) -> None:
    """
    Check whether too much data was transformed to NaN during cleaning.

    Args:
        data_start: Dataframe before cleaning
        data_end: Dataframe after cleaning
        thresh: Max % (in range 0-1) of allowed NaN removed
        td: Tag dictionary

    Raises: ValueError if too much data was removed from a variable.
    """
    n_start = len(data_start)
    nan_start = data_start.isna().sum()
    nan_end = data_end.isna().sum()

    diff = (nan_end - nan_start) / n_start
    filtered_diff = diff[diff > thresh]

    # Determine features that are active in models
    active_features = find_active_features(td)
    filtered_diff = filtered_diff.loc[
        [col for col in filtered_diff.index if col in active_features]
    ]

    if len(filtered_diff) > 0:
        msg = (
            f"The following variables removed more than {thresh * 100}% of data "
            f"during cleaning: \n{100 * filtered_diff.sort_values(ascending=False)}"
        )
        logger.error(msg)
        raise ValueError(msg)


def _clean_data_from_min_max_dict(
    data: pd.DataFrame, min_max_dict: dict
) -> pd.DataFrame:
    """Replace by NaN all values outside min/max range in data.

    Args:
        data: DataFrame for which outliers will be removed.
        min_max_dict: Dictionary of columns and ranges allowed for each variable.

    Returns:
        df: DataFrame with values outside range for the specified
        columns converted to np.nan.

    """
    df = data.copy()
    for col in min_max_dict:
        if col in df:
            try:
                df.loc[
                    (df[col] < min_max_dict[col]["range_min"])
                    | (df[col] > min_max_dict[col]["range_max"]),
                    col,
                ] = np.nan
            except KeyError:
                msg = f"min_max_dict not correctly specified for variable ´{col}´"
                logger.error(msg)
                raise KeyError(msg)
        else:
            logger.warning(f"Column ´{col}´ not found in df during min/max cleaning")

    return df


# All code below is borrowed from the skimpy open source library
# (https://pypi.org/project/skimpy/)
def clean_columns(
    df: pd.DataFrame,
    case: str = "snake",
    replace: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Clean messy column names of a pandas dataframe.
    Args:
        df (pd.DataFrame): Dataframe from which column names are to be cleaned.
        case (str, optional): The desired case style of the column name.
        Defaults to "snake".
                - 'snake': 'column_name'
                - 'kebab': 'column-name'
                - 'camel': 'columnName'
                - 'pascal': 'ColumnName'
                - 'const': 'COLUMN_NAME'
                - 'sentence': 'Column name'
                - 'title': 'Column Name'
                - 'lower': 'column name'
                - 'upper': 'COLUMN NAME'
        replace (Optional[Dict[str, str]], optional): Values to replace
        in the column names.
        Defaults to None.
                - {'old_value': 'new_value'}
    Raises:
        ValueError: If case is not valid.
    Returns:
        pd.DataFrame: Dataframe with cleaned column names.
    """
    if case not in CASE_STYLES:
        raise ValueError(
            f"case {case} is invalid, options are: {', '.join(c for c in CASE_STYLES)}"
        )

    if replace:
        df = df.rename(columns=lambda col: _replace_values(col, replace))

    df = df.rename(columns=lambda col: _convert_case(col))
    df.columns = _rename_duplicates(df.columns, case)

    if replace:  # TODO: hot fix to prevent p80 vs p_80 issue
        df = df.rename(columns=lambda col: col.replace("__p80_", "_p80"))
        df = df.rename(columns=lambda col: col.replace("p_80", "p80"))

    return df


def _rename_duplicates(names: pd.Index, case: str) -> Any:
    """Rename duplicated column names to append a number at the end."""
    if case in {"snake", "const"}:
        sep = "_"
    elif case in {"camel", "pascal"}:
        sep = ""
    elif case == "kebab":
        sep = "-"
    else:
        sep = " "

    names = list(names)
    counts: Dict[str, int] = {}

    for i, col in enumerate(names):
        cur_count = counts.get(col, 0)
        if cur_count > 0:
            names[i] = f"{col}{sep}{cur_count}"
        counts[col] = cur_count + 1

    return names


def _replace_values(name: Any, mapping: Dict[str, str]) -> Any:
    """Replace string values in the column name.

    Parameters
    ----------
    name
        Column name.
    mapping
        Maps old values in the column name to the new values.
    """
    if name in NULL_VALUES:
        return name

    name = str(name)
    for old_value, new_value in mapping.items():
        # If the old value or the new value is not alphanumeric,
        # add underscores to the beginning and end so the new value
        # will be parsed correctly for _convert_case()
        new_val = (
            rf"{new_value}"
            if old_value.isalnum() and new_value.isalnum()
            else rf"_{new_value}_"
        )
        name = re.sub(rf"{old_value}", new_val, name, flags=re.IGNORECASE)

    return name


CASE_STYLES = {
    "snake",
    "kebab",
    "camel",
    "pascal",
    "const",
    "sentence",
    "title",
    "lower",
    "upper",
}

NULL_VALUES = {np.nan, "", None}


def _convert_case(name: Any) -> Any:
    """Convert case style of a column name.

    Args:
        name (Any): Column name.

    Returns:
        Any: name with case converted.
    """
    if name in NULL_VALUES:
        name = "header"

    words = _split_strip_string(str(name))
    name = "_".join(words).lower()

    return name


def _split_strip_string(string: str) -> List[str]:
    """Split the string into separate words and strip punctuation."""
    string = re.sub(r"[!()*+\,\-./:;<=>?[\]^_{|}~]", " ", string)
    string = re.sub(r"[\'\"\`]", "", string)

    return re.sub(
        r"([A-Z][a-z]+)", r" \1", re.sub(r"([A-Z]+|[0-9]+|\W+)", r" \1", string)
    ).split()
