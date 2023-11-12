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
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger(__name__)


class SplitterBase(ABC):
    def split(self, data: pd.DataFrame) -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
        """Split provided data into train and test datasets.

        Args:
            data: DataFrame for splitting

        Returns:
            A tuple with train data and test data
        """
        data = data.copy()
        logger.info(f"Length of data before splitting is {len(data)}")  # noqa: WPS237
        train_data, test_data = self._split(data)
        logger.info(
            "Length of the train data after"  # noqa: WPS237
            f" splitting is {len(train_data)},"
            f" length of the test data"
            f" after splitting is {len(test_data)}.",
        )
        return train_data, test_data

    @abstractmethod
    def __repr__(self) -> str:
        """
        Abstract method for string representation for splitters.

        Notes:
            As per definition of `__repr__` we strive to return a string representation
            that would yield an object with the same value when passed to eval();

        Returns:
            String representation.
        """

    @abstractmethod
    def _split(self, data: pd.DataFrame) -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Abstract method for implementing the split logic for the provided data.

        Returns:
              A tuple with train data and test data
        """
