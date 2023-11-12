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
Data splitting functions.
"""
import typing as tp

from .base_splitter import SplitterBase
from .by_date_splitter import ByDateSplitter
from .by_frac_splitter import ByFracSplitter
from .by_intervals_splitter import ByIntervalsSplitter
from .by_last_window import ByLastWindowSplitter
from .by_sequential_splitter import BySequentialSplitter

SUPPORTED_SPLITTERS: tp.Dict[str, tp.Type[SplitterBase]] = {  # noqa: WPS407
    "date": ByDateSplitter,
    "frac": ByFracSplitter,
    "intervals": ByIntervalsSplitter,
    "last_window": ByLastWindowSplitter,
    "sequential_window": BySequentialSplitter,
}


def create_splitter(
    split_method: str,
    **splitting_parameters: tp.Any,
) -> SplitterBase:
    """
    Create ``SplitterBase`` instance from split_method provided.

    Supported str options for ``split_method``:
        * date to initialize ``ByDateSplitter``
        * frac to initialize ``ByFracSplitter``
        * intervals to initialize ``ByIntervalsSplitter``
        * last_window to initialize ``ByLastWindowSplitter``
        * sequential_window to initialize ``BySequentialSplitter``

    Args:
        split_method: method for choosing type of inheritor of ModelBase to initialize
        splitting_parameters: parameters used for splitter initialization

    Returns:

    """
    if split_method not in SUPPORTED_SPLITTERS:
        supported_splitting_methods = ", ".join(SUPPORTED_SPLITTERS.keys())
        raise ValueError(
            f"Provided splitting method {split_method} is not supported:"
            f" supported splitting methods are {supported_splitting_methods}",
        )
    splitter_type = SUPPORTED_SPLITTERS[split_method]
    return splitter_type(**splitting_parameters)
