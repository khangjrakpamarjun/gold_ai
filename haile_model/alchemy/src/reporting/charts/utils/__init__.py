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
Contains some reusable chart utils.

When adding new utils, try to really keep them REUSABLE.
To avoid creating new dependencies, try to keep utils in the plotting class/module/etc.
"""

from ._chart_operation import (
    add_watermark,
    apply_chart_style,
    cast_color_to_rgba,
    get_default_fig_layout_params,
    get_next_available_legend_group_name,
)
from ._distplot import calculate_optimal_bin_width
from ._error_estimation import get_error_range
from ._validators import check_data
