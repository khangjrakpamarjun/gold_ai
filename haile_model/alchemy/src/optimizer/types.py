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
Holds custom types used throughout the code.
"""

from typing import Union  # pylint: disable=E0611

import numpy as np
import pandas as pd
from typing_extensions import Protocol

Vector = Union[np.ndarray, pd.Series]
Matrix = Union[np.ndarray, np.matrix, pd.DataFrame]


class Predictor(Protocol):
    """
    Protocol type class to define any class with a predict method.
    """

    def predict(self, parameters: Matrix, **kwargs) -> Vector:  # pylint: disable=W0613
        pass
