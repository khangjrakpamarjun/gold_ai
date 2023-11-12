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

import numpy as np
import pandas as pd

from .base_splitter import SplitterBase
from .splitting_utils import convert_datetime, verify_column_inside_dataframe


class BySequentialSplitter(SplitterBase):
    """
    Splits data into blocks of ``block_freq`` size using ``train_freq`` amount of the
    data per block for training, and the rest for testing.

    For example, if ``block_freq`` is "1M" and train_freq is "3W" each month, we will
    use the first three weeks for training and the last week for testing.
    """

    def __init__(
        self,
        datetime_column: str,
        block_freq: str,
        train_freq: str,
        **to_datetime_kwargs: tp.Any,
    ) -> None:
        """
        Args:
            datetime_column: name of ``datetime`` column to use for splitting
            block_freq: size of block to be used for splitting
             provided data and consumable by ``pd.to_timedelta``
            train_freq: size of block used for training
             dataset consumable by ``pd.to_timedelta``
            **to_datetime_kwargs: keyword arguments to pass to pd.to_datetime

        Raises:
            ValueError: if ``block_freq`` is less or equal than ``train_freq``
        """
        self._datetime_column = datetime_column
        self._block_freq = pd.to_timedelta(block_freq)
        self._train_freq = pd.to_timedelta(train_freq)
        self._validate_block_is_greater_than_train_block()
        self._to_datetime_kwargs: tp.Dict[str, tp.Any] = to_datetime_kwargs

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        kwargs_representation = [
            f"{parameter_name}={repr(parameter_value)}"  # noqa: WPS237
            for parameter_name, parameter_value in self._to_datetime_kwargs.items()
        ]
        kwargs_representation_joined = ", ".join(kwargs_representation)
        return (
            f"{class_name}("  # noqa: WPS221,WPS237
            f"datetime_column={repr(self._datetime_column)}, "
            f"block_freq={repr(self._block_freq)}, "
            f"train_freq={repr(self._train_freq)}, "
            f"{kwargs_representation_joined}"
            ")"
        )

    def _split(self, data) -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
        verify_column_inside_dataframe(data, self._datetime_column)
        data[self._datetime_column] = convert_datetime(
            data[self._datetime_column],
            **self._to_datetime_kwargs,
        )
        data_datetime = data[self._datetime_column]
        start_datetime = data_datetime.min()
        end_datetime = data_datetime.max()
        train_mask = np.zeros_like(data_datetime).astype(bool)
        while start_datetime < end_datetime:
            train_mask |= (start_datetime <= data_datetime) & (  # noqa: WPS465
                data_datetime < start_datetime + self._train_freq
            )
            start_datetime += self._block_freq
        return data[train_mask], data[~train_mask]

    def _validate_block_is_greater_than_train_block(self) -> None:
        if self._block_freq <= self._train_freq:
            raise ValueError(
                f"block_freq must be strictly greater than train_freq."
                f" Got block_freq={self._block_freq} <= train_freq={self._train_freq}.",
            )
