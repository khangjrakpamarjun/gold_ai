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
from datetime import datetime

import pandas as pd

from .base_splitter import SplitterBase
from .splitting_utils import (
    convert_datetime,
    verify_column_inside_dataframe,
    verify_split_datetime_inside_series,
)

logger = logging.getLogger(__name__)

TDateTimeScalar = tp.Union[str, float, datetime]


class ByDateSplitter(SplitterBase):
    """
    Split the provided DataFrame into two DataFrames: one with rows strictly before on
    the provided ``split_date`` and one with rows grater or equal to ``split_date``.
    """

    def __init__(
        self,
        datetime_column: str,
        split_datetime: TDateTimeScalar,
        **to_datetime_kwargs: tp.Any,
    ) -> None:
        """
        Args:
            datetime_column: name of ``datetime`` column to use for splitting.
            split_date: date value to split on.
             Can be anything that pd.to_datetime handles.
            to_datetime_kwargs: keyword arguments to pass to pd.to_datetime.
        """
        self._datetime_column: str = datetime_column
        self._split_datetime: pd.Timestamp = convert_datetime(
            split_datetime,
            **to_datetime_kwargs,
        )
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
            f"split_datetime={repr(self._split_datetime)}, "
            f"{kwargs_representation_joined}"
            ")"
        )

    def _split(
        self,
        data: pd.DataFrame,
    ) -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Args:
            data: input DataFrame to split

        Raises:
            ValueError: if ``split_date`` is outside the range of dates in ``data``

            ValueError: if ``split_date`` timezone is misaligned
            with the timezone provided in data

        Returns:
            data before ``split_date``, data after ``split_date``
        """
        verify_column_inside_dataframe(data, self._datetime_column)
        data[self._datetime_column] = convert_datetime(
            data[self._datetime_column],
            **self._to_datetime_kwargs,
        )
        data_datetime = data[self._datetime_column]
        self._verify_timezones_match(data)
        logger.info(f"Splitting by datetime: {self._split_datetime}")
        verify_split_datetime_inside_series(
            data_datetime,
            self._split_datetime,
        )
        train_mask = data_datetime < self._split_datetime
        test_mask = ~train_mask
        return data[train_mask], data[test_mask]

    def _verify_timezones_match(self, data: pd.DataFrame) -> None:
        if data[self._datetime_column].dt.tz != self._split_datetime.tz:
            split_datetime_timezone = self._split_datetime.tz
            provided_data_timezone = data[self._datetime_column].dt.tz
            raise ValueError(
                "Provided timezone for in split_datetime"
                " doesn't match timezone for data: split_date timezone is"
                f" {split_datetime_timezone}, data timezone"
                f" is {provided_data_timezone}",
            )
