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
optimus_core
Stand-alone package for core optimus-related tasks
"""

__version__ = "0.17.3"


from .metrics import mean_absolute_percentage_error
from .pi_connector import (
    BaseOAIConnector,
    OAICurrentValueStreams,
    OAIRecordedStreams,
    OAISummaryStreams,
    convert_timezone,
    get_current_time,
    round_minutes,
)
from .tag_dict import TagDict
from .transformer import (
    ColumnNamesAsNumbers,
    DropAllNull,
    DropColumns,
    NumExprEval,
    SelectColumns,
    SkLearnSelector,
    SklearnTransform,
    Transformer,
)
from .utils import generate_run_id, load_obj, partial_wrapper
