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
from abc import ABC, abstractmethod

import pandas as pd


class EvaluatesMetrics(ABC):
    @abstractmethod
    def evaluate_metrics(
        self,
        data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> tp.Dict[str, float]:
        """
        Define API for objects, that produce evaluation
        metrics based on the provided data

        Args:
            data: data to calculate metrics
            **kwargs: additional keyword arguments that
             are required for method implementation

        Returns:
            Mapping from metric name into metric value
        """
