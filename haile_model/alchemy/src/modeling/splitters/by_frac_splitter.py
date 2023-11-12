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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .base_splitter import SplitterBase
from .splitting_utils import convert_datetime, verify_column_inside_dataframe

logger = logging.getLogger(__name__)


class ByFracSplitter(SplitterBase):
    """
    Split the data by desired train/test fraction. Uses
    ``sklearn.model_selection.train_test_split`` under the hood.
    """

    def __init__(
        self,
        datetime_column: tp.Optional[str] = None,
        sort: bool = False,
        test_size: tp.Optional[float] = None,
        train_size: tp.Optional[float] = None,
        random_state: tp.Optional[int] = None,
        shuffle: bool = False,
        stratify: tp.Optional[np.array] = None,
        **to_datetime_kwargs: tp.Any,
    ) -> None:
        """
        Args:
            datetime_column: optional column name to sort by if ``sort`` true
            sort: optional, if true, will sort by ``datetime_column``
            test_size: parameter to pass into ``sklearn.train_test_split``
            train_size: parameter to pass into ``sklearn.train_test_split``
            random_state: parameter to pass into ``sklearn.train_test_split``
            shuffle: parameter to pass into ``sklearn.train_test_split``
            stratify: parameter to pass into ``sklearn.train_test_split``
            to_datetime_kwargs: keyword arguments to pass to pd.to_datetime

        Raises:
            ValueError: if ``sort`` passed as True and at
            the same time ``shuffle`` passed as True
            and ``stratify`` is non-trivial.

            ValueError: if ``sort`` passed as True
            and ``datetime_column`` is not specified
        """
        self._datetime_column = datetime_column
        self._sort = sort
        self._to_datetime_kwargs = to_datetime_kwargs
        self._train_test_split_kwargs = {
            "shuffle": shuffle,
            "stratify": stratify,
            "random_state": random_state,
            "train_size": train_size,
            "test_size": test_size,
        }
        self._validate_init_parameters()

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        datetime_kwargs_repr = [
            f"{parameter_name}={repr(parameter_value)}"  # noqa: WPS237
            for parameter_name, parameter_value in self._to_datetime_kwargs.items()
        ]
        datetime_kwargs_repr_joined = ", ".join(datetime_kwargs_repr)
        train_test_split_repr = [
            f"{parameter_name}={repr(parameter_value)}"  # noqa: WPS237
            for parameter_name, parameter_value in self._train_test_split_kwargs.items()
        ]
        train_test_split_repr_joined = ", ".join(train_test_split_repr)
        return (
            f"{class_name}("  # noqa: WPS221,WPS237
            f"datetime_column={repr(self._datetime_column)}, "  # noqa: WPS221
            f"sort={repr(self._sort)}, "
            f"{train_test_split_repr_joined}, "
            f"{datetime_kwargs_repr_joined}"
            ")"
        )

    def _split(self, data) -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
        if self._sort:
            data = self._sort_by_date(data)
        return train_test_split(data, **self._train_test_split_kwargs)

    def _sort_by_date(self, data: pd.DataFrame) -> pd.DataFrame:
        if self._datetime_column is None:
            raise ValueError(
                "Must provide datetime_column if sort_date" " is `True`. Got `None`.",
            )
        verify_column_inside_dataframe(data, self._datetime_column)
        data[self._datetime_column] = convert_datetime(
            data[self._datetime_column],
            **(self._to_datetime_kwargs or {}),
        )
        return data.sort_values(by=self._datetime_column)

    def _validate_init_parameters(self):
        if self._sort:
            conflict_condition = (
                self._train_test_split_kwargs["stratify"] is not None
                or self._train_test_split_kwargs["shuffle"]
            )
            if conflict_condition:
                raise ValueError(
                    "Splitting with stratification or shuffling is not"
                    " possible when data sorting is enabled.",
                )
            if self._datetime_column is None:
                raise ValueError(
                    "Datetime column for data is not provided."
                    " Use parameter `datetime_column` to specify column for datetime.",
                )
