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
Constraint handling package.
"""

from .constraint import (  # noqa: F401
    EqualityConstraint,
    InequalityConstraint,
    SetMembershipConstraint,
    constraint,
)
from .helpers import make_boundary_penalties  # noqa: F401
from .penalty import Penalty, history_penalty, penalty  # noqa: F401
from .repair import Repair, SetRepair, UserDefinedRepair, repair  # noqa: F401
