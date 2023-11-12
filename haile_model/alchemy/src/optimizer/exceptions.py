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
Holds custom types used throughout the code.
"""


class InitializationError(Exception):
    """
    Error used to indicate something going wrong during initialization of a solver.
    """


class MaxIterationError(Exception):
    """
    Error used for reaching maximum allowed iterations in a certain algorithm.
    """


class InvalidConstraintError(Exception):
    """
    Error used to indicate a constraint is improperly defined.
    """


class InvalidObjectiveError(Exception):
    """
    Error used to indicate an invalid objective.
    """


class SolutionNotFoundError(Exception):
    """
    Error when solver does not converge/solutions are not found.
    """
