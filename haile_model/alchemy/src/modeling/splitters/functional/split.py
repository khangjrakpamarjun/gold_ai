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

from ..base_splitter import SplitterBase


def split_data(
    data: pd.DataFrame,
    splitter: SplitterBase,
) -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data using the provided splitter.

    Args:
        data: input data to split.
        splitter: instance of ``optimus_core.SplitterBase``

    Returns:
        (train, test) datasets.
    """
    return splitter.split(data)
