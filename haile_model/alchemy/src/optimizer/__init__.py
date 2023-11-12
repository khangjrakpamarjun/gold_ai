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
optimizer
Stand-alone optimization package
"""

__version__ = "2.8.12"

from optimizer.constraint.penalty import Penalty, penalty  # noqa: F401
from optimizer.constraint.repair import (  # noqa: F401
    Repair,
    SetRepair,
    UserDefinedRepair,
    repair,
)
from optimizer.problem import (  # noqa: F401
    DAGOptimizationProblem,
    OptimizationProblem,
    StatefulOptimizationProblem,
)
