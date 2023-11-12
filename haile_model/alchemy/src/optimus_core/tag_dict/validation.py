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
Tag Dict Validation
"""

import typing as tp

import pandas as pd

REQUIRED_COLUMNS = frozenset(
    (
        "tag",
        "name",
        "tag_type",
        "data_type",
        "unit",
        "range_min",
        "range_max",
        "on_off_dependencies",
        "derived",
    )
)

UNIQUE = frozenset(("tag", "name"))

COMPLETE = frozenset(("tag",))

KNOWN_VALUES = {  # noqa: WPS407
    "tag_type": {"input", "output", "state", "control", "on_off"},
    "data_type": {"numeric", "categorical", "boolean", "datetime", "numeric_coerce"},
}

# tags are checked for whether they break any of the below rules
# captured as rule - explanation
ILLEGAL_TAG_PATTERNS = (
    ("^.*,+.*$", "no commas in tag"),
    (r"^\s.*$", "tag must not start with whitespace character"),
    (r"^.*\s$", "tag must not end with whitespace character"),
)


class TagDictError(Exception):
    """Tag Dictionary related exceptions"""


def validate_td(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validates a tag dict dataframe

    Returns:
        validated dataframe with comma separated values parsed to lists
    """
    data = data.copy()
    _check_required_columns_are_present(data)
    _check_complete_columns_has_no_missing_values(data)
    _check_unique_columns_contain_no_duplicates(data)
    _check_tags_do_not_contain_invalid_characters(data)
    _check_all_restricted_values_are_valid(data)
    _parse_on_off_dependencies(data)
    _check_on_off_dependencies(data)
    return data


def _check_on_off_dependencies(data: pd.DataFrame) -> None:
    all_tags = set(data["tag"])
    on_off_tags = set(data.loc[data["tag_type"] == "on_off", "tag"])
    for idx, deps in data["on_off_dependencies"].items():
        _check_on_off_dependencies_tags_exist(all_tags, data, deps, idx)
        _check_dependencies_are_of_on_off_type(data, deps, idx, on_off_tags)


def _parse_on_off_dependencies(data: pd.DataFrame) -> None:
    if isinstance(data["on_off_dependencies"].iloc[0], list):
        return
    data["on_off_dependencies"] = (
        data["on_off_dependencies"]
        .fillna("")
        .astype(str)
        .str.split(",")
        .apply(
            lambda dependency_tags: [
                tag.strip() for tag in dependency_tags if tag.strip()
            ],
        )
    )


def _check_dependencies_are_of_on_off_type(
    data: pd.DataFrame,
    deps: tp.List[str],
    idx: tp.Any,
    on_off_tags: tp.Set[str],
) -> None:
    unknown_on_off_tags = set(deps) - on_off_tags
    if unknown_on_off_tags:
        tag_name = data.loc[idx, "tag"]
        raise TagDictError(
            f"The following on_off_dependencies of {tag_name} "
            f"are not labelled as on_off type tags: {unknown_on_off_tags}",
        )


def _check_on_off_dependencies_tags_exist(
    all_tags: tp.Set[str],
    data: pd.DataFrame,
    deps: tp.List[str],
    idx: tp.Any,
) -> None:
    unknown_tags = set(deps) - all_tags
    if unknown_tags:
        tag_name = data.loc[idx, "tag"]
        raise TagDictError(
            f"The following on_off_dependencies of {tag_name} "
            f"are not known tags: {unknown_tags}",
        )


def _check_all_restricted_values_are_valid(data: pd.DataFrame) -> None:
    for col, known_vals in KNOWN_VALUES.items():
        invalid = set(data[col].dropna()) - known_vals
        if invalid:
            raise TagDictError(
                f"Found invalid entries in column {col}: {invalid}. "
                f"Must be one of: {known_vals}",
            )


def _check_tags_do_not_contain_invalid_characters(data: pd.DataFrame) -> None:
    for pattern, rule in ILLEGAL_TAG_PATTERNS:
        matches = data.loc[data["tag"].str.match(pattern), "tag"]
        if not matches.empty:
            formatted_matches = ", ".join(matches)
            raise TagDictError(
                f"The following tags don't adhere to rule `{rule}`:"
                f" {formatted_matches}",
            )


def _check_unique_columns_contain_no_duplicates(data: pd.DataFrame) -> None:
    for col in UNIQUE:
        duplicates = data.loc[data[col].duplicated(), col]
        if not duplicates.empty:
            formatted_duplicates = ", ".join(duplicates)
            raise TagDictError(
                "The following values are duplicated"
                f" in column `{col}`: {formatted_duplicates}",
            )


def _check_complete_columns_has_no_missing_values(data: pd.DataFrame) -> None:
    for col in COMPLETE:
        if data[col].isnull().any():
            raise TagDictError(f"Found missing values in column `{col}`")


def _check_required_columns_are_present(data: pd.DataFrame) -> None:
    missing_cols = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing_cols:
        raise TagDictError(
            "The following columns are missing from"
            f" the input dataframe: {missing_cols}",
        )
