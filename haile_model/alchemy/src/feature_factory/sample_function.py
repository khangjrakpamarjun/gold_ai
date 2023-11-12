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
"""Schemas required by pydantic to validate the format of the configuration"""

from typing import Iterable

import pandas as pd


def pandas_mean(data: pd.DataFrame, dependencies: Iterable[str], **kwargs) -> pd.Series:
    """Wrapper function to run pandas.DataFrame.mean()

    Args:
        data (pd.DataFrame): data
        dependencies (Iterable[str]): list of columns for input
        **kwargs: kwargs for pandas.DataFrame.mean()

    Returns:
        pd.Series: calculated result
    """
    return data[dependencies].mean(axis=1, **kwargs)


def pandas_prod(data: pd.DataFrame, dependencies: Iterable[str], **kwargs) -> pd.Series:
    """Wrapper function to run pandas.DataFrame.prod()

    Args:
        data (pd.DataFrame): data
        dependencies (Iterable[str]): list of columns for input
        **kwargs: kwargs for pandas.DataFrame.prod()

    Returns:
        pd.Series: calculated result
    """
    return data[dependencies].prod(axis=1, **kwargs)


def pandas_sum(data: pd.DataFrame, dependencies: Iterable[str], **kwargs) -> pd.Series:
    """Wrapper function to run pandas.DataFrame.sum()

    Args:
        data (pd.DataFrame): data
        dependencies (Iterable[str]): list of columns for input
        **kwargs: kwargs for pandas.DataFrame.sum()

    Returns:
        pd.Series: calculated result
    """
    return data[dependencies].sum(axis=1, **kwargs)


def pandas_max(data: pd.DataFrame, dependencies: Iterable[str], **kwargs) -> pd.Series:
    """Wrapper function to run pandas.DataFrame.max()

    Args:
        data (pd.DataFrame): data
        dependencies (Iterable[str]): list of columns for input
        **kwargs: kwargs for pandas.DataFrame.max()

    Returns:
        pd.Series: calculated result
    """
    return data[dependencies].max(axis=1, **kwargs)


def pandas_min(data: pd.DataFrame, dependencies: Iterable[str], **kwargs) -> pd.Series:
    """Wrapper function to run pandas.DataFrame.min()

    Args:
        data (pd.DataFrame): data
        dependencies (Iterable[str]): list of columns for input
        **kwargs: kwargs for pandas.DataFrame.min()

    Returns:
        pd.Series: calculated result
    """
    return data[dependencies].min(axis=1, **kwargs)


def pandas_divide(
    data: pd.DataFrame,
    dependencies: Iterable[str],
    **kwargs,
) -> pd.Series:
    """Wrapper function to run pandas.DataFrame.divide()

    Args:
        data (pd.DataFrame): data
        dependencies (Iterable[str]): list of columns for input
        **kwargs: kwargs for pandas.DataFrame.divide()

    Raises:
        ValueError: if dependencies isn't a list of 2 columns

    Returns:
        pd.Series: calculated result
    """
    dependencies_count = len(dependencies)
    if dependencies_count != 2:
        raise ValueError(
            "`dependencies` should contain two"
            f" variable names: given {dependencies_count}.",
        )
    numerator = dependencies[0]
    denominator = dependencies[1]
    return data[numerator].div(data[denominator], **kwargs)


def pandas_subtract(
    data: pd.DataFrame,
    dependencies: Iterable[str],
    **kwargs,
) -> pd.Series:
    """Wrapper function to run pandas.DataFrame.subtract()

    Args:
        data (pd.DataFrame): data
        dependencies (Iterable[str]): list of columns for input
        **kwargs: kwargs for pandas.DataFrame.subtract()

    Raises:
        ValueError: if dependencies isn't a list of 2 columns

    Returns:
        pd.Series: calculated result
    """
    dependencies_count = len(dependencies)
    if dependencies_count != 2:
        raise ValueError(
            "`dependencies` should contain two variable"
            f" names: given {dependencies_count}.",
        )
    from_col = dependencies[0]
    to_col = dependencies[1]

    return data[from_col].subtract(data[to_col], **kwargs)
