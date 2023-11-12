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
Contains basic reusable charts.
Those can be used for creating standalone visuals
or extended to more complex charts/dashboards.
"""

from ._code import CodeFormattingError, CodePlot, TFormatter, plot_code  # noqa: F401
from ._correlation import plot_correlation  # noqa: F401
from ._line import draw_overlay, draw_range, plot_lines  # noqa: F401
from ._string import plot_string  # noqa: F401
from ._table import TablePlot, plot_table  # noqa: F401
from ._wrapped_builtins import (  # noqa: F401
    plot_bar,
    plot_box,
    plot_histogram,
    plot_scatter,
    plot_timeline,
)
