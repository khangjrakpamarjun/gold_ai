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
import typing as tp

import pandas as pd

from .base_splitter import SplitterBase
from .splitting_utils import (
    convert_datetime,
    verify_column_inside_dataframe,
    verify_split_datetime_inside_series,
)

logger = logging.getLogger(__name__)


class ByLastWindowSplitter(SplitterBase):
    """
    Splits data into everything before and after the provided last window.
    For example if ``freq="1M"``, the return values will be all the data excluding the
    most recent month and the most recent month of data.

    See here for a list of offset aliases:
        pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    """

    def __init__(
        self,
        datetime_column: str,
        freq: str,
        **to_datetime_kwargs: tp.Any,
    ) -> None:
        self._datetime_column = datetime_column
        self._freq = freq
        self._to_datetime_kwargs: tp.Dict[str, tp.Any] = to_datetime_kwargs

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        kwargs_representation = [
            f"{parameter_name}={repr(parameter_value)}"  # noqa: WPS237
            for parameter_name, parameter_value in self._to_datetime_kwargs.items()
        ]
        kwargs_representation_joined = ", ".join(kwargs_representation)
        return (
            f"{class_name}("  # noqa: WPS237
            f"datetime_column={repr(self._datetime_column)}, "
            f"freq={repr(self._freq)}, "
            f"{kwargs_representation_joined}"
            ")"
        )

    def _split(self, data) -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Args:
            data: input DataFrame to split

        Raises:
            ValueError: if provided ``freq`` is larger than
             the timeline provided in the data.

        Returns:
            data before last window, data in last window
        """
        verify_column_inside_dataframe(data, self._datetime_column)
        data[self._datetime_column] = convert_datetime(
            data[self._datetime_column],
            **self._to_datetime_kwargs,
        )
        data_datetime = data[self._datetime_column]
        split_datetime = data_datetime.max() - pd.tseries.frequencies.to_offset(
            self._freq
        )
        verify_split_datetime_inside_series(data_datetime, split_datetime)
        logger.info(f"Splitting by datetime: {split_datetime}")
        train_mask = data_datetime <= split_datetime
        test_mask = ~train_mask
        return data[train_mask], data[test_mask]
