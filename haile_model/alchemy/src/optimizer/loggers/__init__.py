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
Logger package.
"""
__all__ = [
    "BasicLogger",
    "PenaltyLogger",
    "NBestLogger",
    "BasicFileLogger",
    "PenaltyFileLogger",
]

from contextlib import suppress

from optimizer.loggers.basic import BasicLogger  # noqa: F401
from optimizer.loggers.best_trajectory import BestTrajectoryLogger  # noqa: F401
from optimizer.loggers.n_best import NBestLogger  # noqa: F401
from optimizer.loggers.penalty import PenaltyLogger  # noqa: F401

with suppress(ImportError):
    from optimizer.loggers.penalty_file import PenaltyFileLogger  # noqa: F401

with suppress(ImportError):
    from optimizer.loggers.basic_file import BasicFileLogger  # noqa: F401
