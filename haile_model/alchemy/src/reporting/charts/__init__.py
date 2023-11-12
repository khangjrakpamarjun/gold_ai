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
The ``charts`` module contains plotting functions that produce
package compatible figures (see ``api.types.PlotlyLike``, ``api.MatplotlibLike``).
"""

from . import utils
from ._feature_overview import plot_feature_overviews
from .batchplot import *
from .modeling import *  # noqa: WPS440
from .primitive import *  # noqa: WPS440
