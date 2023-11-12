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

"""
Validation utility functions.
"""

from typing import Union

import numpy as np
import pandas as pd

from optimizer.types import Matrix, Vector


def check_matrix(data: Union[Matrix, Vector]):
    """Ensure the provided data is 2-D and throw a helpful error if not.

    Args:
        data: data to check for dimensionality.

    Raises:
        ValueError: when the provided data is one dimensional.
    """
    # Source: sklearn.utils.validation.check_array
    if isinstance(data, pd.Series):
        raise ValueError(
            f"Expected DataFrame, got a Series instead: \nseries={data}.\n"
            f"Convert your data to a DataFrame using pd.DataFrame(series) if "
            f"your data has a single feature or series.to_frame().T "
            f"if it contains a single sample."
        )

    if isinstance(data, np.ndarray):
        if data.ndim == 0 or data.ndim == 1:
            raise ValueError(
                f"Expected 2D array, got scalar array instead:\narray={data}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                f"your data has a single feature or array.reshape(1, -1) "
                f"if it contains a single sample."
            )

        if data.ndim >= 3:
            raise ValueError(f"Found array with dim {data.ndim}. Expected <= 2.")


def check_vector(data: Union[Matrix, Vector]):
    """Ensure the provided data is 1-D and throw a helpful error if not.

    Args:
        data: data to check for dimensionality.

    Raises:
        ValueError: when the provided data is not one dimensional.
    """
    if data.squeeze().ndim > 1:  # Works for Pandas and Numpy.
        raise ValueError(
            f"Expected 1D Vector, got {type(data).__name__} with shape {data.shape}"
        )
