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


import json
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    Helper class to deal with numeric values from Numpy and Pandas.
    """

    def default(self, o: Any) -> Any:  # pylint: disable=method-hidden
        """Overridden default method.
        Deals with converting unserializable numeric types to builtin types.

        Args:
            o: Any object.

        Returns:
            Any.
        """
        if isinstance(o, np.generic) and np.isreal(o):
            return o.item()

        return super(NumpyEncoder, self).default(o)
