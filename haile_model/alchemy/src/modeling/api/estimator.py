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


from __future__ import annotations

import typing as tp

import numpy as np
import numpy.typing as npt
import pandas as pd

Vector = tp.Union[npt.NDArray["np.generic"], pd.Series]
Matrix = tp.Union[npt.NDArray["np.generic"], np.matrix, pd.DataFrame]


class Estimator(tp.Protocol):
    """
    Protocol type class to define scikit-learn compatible estimator.
    """

    def fit(
        self,
        data: Matrix,
        target: Vector,
        **kwargs: tp.Any,
    ) -> Estimator:
        """
        Interface definition for `.fit` aligned with sklearn API
        """

    def predict(
        self,
        data: Matrix,
        **kwargs: tp.Any,
    ) -> np.array:
        """
        Interface definition for `.predict` aligned with sklearn API
        """

    def set_params(  # noqa: WPS615
        self,
        **params: tp.Any,
    ) -> Estimator:
        """
        Interface definition for `.set_params` aligned with sklearn API
        """

    def get_params(self, deep=True) -> tp.Dict[str, tp.Any]:  # noqa: WPS615
        """
        Interface definition for `.get_params` aligned with sklearn API
        """
