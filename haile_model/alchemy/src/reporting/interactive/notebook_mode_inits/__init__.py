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
Sometimes plots doesn't render in the way they are supposed to due to missing CSS/JS.
Functions from this module help to activate
notebook mode (import all missing CSS/JS components).
"""

from ._general_init import (  # noqa: F401
    SUPPORTED_INIT_TARGETS,
    init_notebook_mode,
    init_notebook_mode_for_code,
)
