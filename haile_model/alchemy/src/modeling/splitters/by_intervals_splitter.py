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
from pandas.core.tools.datetimes import DatetimeScalar

from .base_splitter import SplitterBase
from .splitting_utils import convert_datetime, verify_column_inside_dataframe

TDateTimeRange = tp.Tuple[DatetimeScalar, DatetimeScalar]


class ByIntervalsSplitter(SplitterBase):
    """
    Split data based on provided intervals.
    Intervals are considered right-open, meaning that if
    [start_datetime, `end_datetime`) is provided,
    then `end_datetime` is not considered as a part of interval.

    * If only ``train_periods`` are provided, all other data will be used for testing.
    * If only ``test_periods`` are provided, all other data will be used for training.
    * If both ``train_periods`` and ``test_periods`` provided,
     samples outside the ranges will not be used.

    Intervals are required to be in the format: ``[start_date, end_date)``. Where both
    values can be anything that pd.to_datetime can handle.
    """

    def __init__(
        self,
        datetime_column: str,
        train_intervals: tp.Optional[tp.List[TDateTimeRange]] = None,
        test_intervals: tp.Optional[tp.List[TDateTimeRange]] = None,
        strict: bool = True,
        **to_datetime_kwargs: tp.Any,
    ) -> None:
        """
        Args:
            datetime_column: name of ``datetime`` column to use for splitting
            train_intervals: List of tuples with length 2, representing start
             and end of the intervals used for train dataset
            test_intervals: List of tuples with length 2, representing start
             and end of the intervals used for test dataset
            strict: optional flag, True enforces train and test periods do not overlap
            **to_datetime_kwargs: keyword arguments to pass to pd.to_datetime

        Raises:
            ValueError: if any of the provided periods for train
            or test has length different from 2

            ValueError: if start datetime is later than end datetime
            for any of the intervals  provided for train ot test

            ValueError: if none of the train or test intervals provided

            ValueError: if ``strict`` parameter is set as True
             and train and test intervals intersect
        """
        self._datetime_column = datetime_column
        self._train_intervals = (
            _validate_intervals_correct(train_intervals)
            if train_intervals is not None
            else None
        )
        self._test_intervals = (
            _validate_intervals_correct(test_intervals)
            if test_intervals is not None
            else None
        )
        self._strict = strict
        self._to_datetime_kwargs: tp.Dict[str, tp.Any] = to_datetime_kwargs
        self._validate_init_parameters()
        self._validate_intervals_intersection()

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        datetime_kwargs_repr = [
            f"{parameter_name}={repr(parameter_value)}"  # noqa: WPS237
            for parameter_name, parameter_value in self._to_datetime_kwargs.items()
        ]
        datetime_kwargs_repr_joined = ", ".join(datetime_kwargs_repr)
        return (
            f"{class_name}("  # noqa: WPS221,WPS237
            f"datetime_column={repr(self._datetime_column)}, "
            f"train_intervals={repr(self._train_intervals)}, "
            f"test_intervals={repr(self._test_intervals)}, "
            f"strict={repr(self._strict)}, "
            f"{datetime_kwargs_repr_joined}"
            ")"
        )

    def _split(self, data: pd.DataFrame) -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
        verify_column_inside_dataframe(data, self._datetime_column)
        data[self._datetime_column] = convert_datetime(
            data[self._datetime_column],
            **self._to_datetime_kwargs,
        )
        datetime_data = data[self._datetime_column]
        if self._train_intervals is None:
            test_mask = _get_mask_for_data_inside_intervals(
                datetime_data,
                intervals=self._test_intervals,
            )
            return data[~test_mask], data[test_mask]
        if self._test_intervals is None:
            train_mask = _get_mask_for_data_inside_intervals(
                datetime_data,
                intervals=self._train_intervals,
            )
            return data[train_mask], data[~train_mask]
        train_mask = _get_mask_for_data_inside_intervals(
            datetime_data,
            intervals=self._train_intervals,
        )
        test_mask = _get_mask_for_data_inside_intervals(
            datetime_data,
            intervals=self._test_intervals,
        )
        return data[train_mask], data[test_mask]

    def _validate_init_parameters(self) -> None:
        if self._train_intervals is None and self._test_intervals is None:
            raise ValueError(
                "Must provide at least one" " of train intervals or test intervals.",
            )

    def _validate_intervals_intersection(self) -> None:
        if not self._strict:
            return
        if self._train_intervals is None:
            return
        if self._test_intervals is None:
            return
        # IntervalArray.overlaps is not yet implemented
        # when parameter other is IntervalArray
        # https://github.com/pandas-dev/pandas/blob/v1.5.3/pandas/core/arrays/interval.py#L1324
        for train_interval in self._train_intervals:
            for test_interval in self._test_intervals:
                if train_interval.overlaps(test_interval):
                    raise ValueError(
                        f"Provided train intervals {train_interval}"
                        f" and test interval {test_interval} are overlapping."
                        " If this is the desired behavior, set strict=False.",
                    )


def _validate_intervals_correct(
    intervals: tp.List[TDateTimeRange],
    **to_datetime_kwargs: tp.Any,
) -> tp.List[pd.Interval]:
    validated_intervals = []
    for interval in intervals:
        interval_length = len(interval)
        if interval_length != 2:
            raise ValueError(
                "Intervals can only have two values. Got interval"
                f" with length {interval_length}: {interval}",
            )
        start_of_interval, end_of_interval = [
            convert_datetime(datetime, **to_datetime_kwargs) for datetime in interval
        ]
        try:
            validated_interval = pd.Interval(start_of_interval, end_of_interval)
        except ValueError as error:
            raise ValueError(
                "Intervals must be start, end date pairs."
                f" Got {start_of_interval} and {end_of_interval}."
                " Start must be less than end.",
            ) from error
        validated_intervals.append(validated_interval)
    return validated_intervals


def _get_mask_for_data_inside_intervals(
    data_datetime: pd.Series,
    intervals: tp.List[pd.Interval],
) -> np.array:
    mask = np.zeros_like(data_datetime, dtype=bool)
    for interval in intervals:
        mask |= (data_datetime >= interval.left) & (  # noqa: WPS465
            data_datetime < interval.right
        )  # noqa: WPS465
    return mask
