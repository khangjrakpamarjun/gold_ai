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
Holds penalty functions.
"""


__all__ = ["linear", "quadratic"]


def linear(bound_distance):
    """Linear constraint penalty.

    Args:
        bound_distance: Vector, non-negative vector of distances.

    Returns:
        Vector, equal to input vector, no change made.
    """
    return bound_distance


def quadratic(bound_distance):
    """Quadratic constraint penalty.
    Penalty increases quadratically as distance increases.

    Args:
        bound_distance: Vector, non-negative vector of distances.

    Returns:
        Vector, penalties assigned to these distances.
    """
    return bound_distance**2
