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

import typing as tp

import pandas as pd

TStrOrListOfStr = tp.Union[str, tp.List[str]]


def check_data(
    data: pd.DataFrame,
    *requested_columns: tp.Optional[TStrOrListOfStr],
    validate_dataset: bool = True,
    validate_columns: bool = True,
) -> None:
    """
    Validates dataset and columns:
        * Dataset
            - dataset is of `pandas.DataFrame` type
            - dataset is not empty
        * Columns
            - at least one column is requested
            - all requested columns are present in data

    Args:
        data: dataset to validate
        *requested_columns: columns to validate
        validate_dataset: validates dataset if set true
        validate_columns: validates columns if set true

    Raises:
        ValuesError in case any condition is violated
    """
    if validate_dataset:
        _validate_dataset(data)
    if validate_columns:
        _validate_columns(data, *requested_columns)


def _get_requested_columns_set(
    requested_columns: tp.Tuple[tp.Optional[TStrOrListOfStr]],
) -> tp.Set[str]:
    not_none_columns = set()
    for column in requested_columns:
        if column is None:
            continue
        if isinstance(column, str):
            not_none_columns.add(column)
        else:
            not_none_columns.update(column)
    return not_none_columns


def _validate_columns(
    data: pd.DataFrame,
    *requested_columns: tp.Optional[TStrOrListOfStr],
) -> None:
    all_columns_are_none = all(column is None for column in requested_columns)
    if all_columns_are_none:
        raise ValueError("Please provide variable names for reporting")
    not_none_columns = _get_requested_columns_set(requested_columns)
    not_included_cols = set(not_none_columns) - set(data.columns)
    if not_included_cols:
        raise ValueError(
            f"The following columns are missing from the dataframe: "
            f"{not_included_cols}",
        )


def _validate_dataset(data: pd.DataFrame) -> None:
    if isinstance(data, pd.Series):
        raise ValueError("Only `pandas.DataFrame` input type is acceptable")
    if data.empty:
        raise ValueError("Nothing to visualize - dataframe is empty")
