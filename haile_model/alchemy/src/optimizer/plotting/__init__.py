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
Plotting package.
"""

from optimizer.plotting.constraint import penalty_plot  # noqa: F401
from optimizer.plotting.convergence import convergence_plot  # noqa: F401
from optimizer.plotting.trajectory import (  # noqa: F401
    best_trajectory_plot,
    best_trajectory_problem_plot,
)
