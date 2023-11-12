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
Experimental Optimization Components
The `optimizer.experimental` module contains importable __experimental__
modules. These modules include unit-tests. As they mature, they are
subject to change and/or migrate out of this folder without warning.
"""
import logging

from optimizer.experimental.hss.neighbourhood_calcs import (  # noqa: F401
    NeighbourhoodCalculator,
)
from optimizer.experimental.solvers.monte_carlo_solver import (  # noqa: F401
    MonteCarloSolver,
)

logger = logging.getLogger(__name__)
logger.log(
    level=30,
    msg="WARNING: You are importing features from the experimental"
    "module within `optimizer`. As these features mature "
    "they may be changed or deprecated without warning.",
)
