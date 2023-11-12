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

TVector = tp.Union[pd.Series, np.ndarray]
TMatrix = tp.Union[npt.NDArray["np.generic"], np.matrix, pd.DataFrame]


# TODO: Remove this warning when not needed anymore
#  WARNING: Docstring prepared for future implementation
#  Not entirely accurate right now.
class Estimator(tp.Protocol):
    """

    Intended to be compatible (or ideally identical) to ``modeling.Estimator``.

    Protocol type class to define any scikit-learn compatible estimator.
    """

    def fit(
        self,
        dataset: TMatrix,
        target: TVector,
        **kwargs: tp.Any,
    ) -> tp.Any:
        """
        Interface definition  for `.fit` aligned with sklearn API
        """

    def predict(
        self,
        dataset: TMatrix,
        **kwargs: tp.Any,
    ) -> TVector:
        """
        Interface definition  for `.predict` aligned with sklearn API
        """

    def set_params(  # noqa: WPS615
        self,
        **params: tp.Any,
    ) -> tp.Dict[str, tp.Any]:
        """
        Interface definition  for `.set_params` aligned with sklearn API
        """

    def get_params(self, deep=True) -> Estimator:  # noqa: WPS615
        """
        Interface definition  for `.get_params` aligned with sklearn API
        """


class ShapExplanation(tp.Protocol):
    # TODO: Remove this warning when not needed anymore
    #  WARNING: Docstring prepared for future implementation
    #  Not entirely accurate right now.
    """
    Intended to be compatible (or ideally identical) to the ``ShapExplanation`` protocol
    in ``modeling``.

    Type stub protocol for ``shap.Explanation``
    """

    # Using ``values`` to keep the same interface as shap.Explanations
    @property
    def values(self) -> np.array:  # noqa: WPS110
        """
        ``np.array`` of SHAP values
        Returns:

        """

    @property
    def data(self) -> np.array:
        """
        ``np.array`` of original data called to be explained
        """

    @property
    def base_values(self) -> np.array:
        """
        ``np.array`` of SHAP base values - E(F(X)).
        """

    @property
    def feature_names(self) -> np.array:
        """
        ``np.array`` of column names from data
        """
