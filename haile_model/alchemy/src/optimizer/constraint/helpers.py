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
Constraint helper functions.
"""

from typing import List, Tuple, Union

from optimizer.constraint.constraint import InequalityConstraint
from optimizer.constraint.penalty import Penalty
from optimizer.utils.functional import column


def make_boundary_penalties(
    bounds: List[Tuple], columns: List[Union[str, int]]
) -> List[Penalty]:
    """Make a list of Penalties describing the boundary of a problem.

    This function OR the boundary argument to a Solver should be used to enforce
    boundary conditions. When using both, the behavior will be the same as with solver
    boundaries, only, since Solvers will usually clip values.

    Use this when your boundary conditions seem to be overly restrictive or you're using
    a sampling based solver like CMA-Evolutionary Strategies. Also consider tightening
    these boundary penalties over time for improved results.

    This will output 2 * len(bounds) penalties each with the name
    <column>_lower and <column>_upper.

    Args:
        bounds: list of tuples describing the boundaries of the problem.
        columns: list of columns corresponding to each penalty.

    Returns:
        List of Penalties.

    Raises:
        ValueError: if bounds and columns have different lengths.
    """
    # If provided, each boundary should have a name.
    if len(bounds) != len(columns):
        raise ValueError(
            f"Each boundary must have a column. "
            f"Currently given {len(bounds)} boundaries and {len(columns)} columns."
        )

    penalties = []

    for (lo, hi), col in zip(bounds, columns):
        func = column(col)

        penalties += [
            Penalty(InequalityConstraint(func, lower_bound=lo, name=f"{col}_lower")),
            Penalty(InequalityConstraint(func, upper_bound=hi, name=f"{col}_upper")),
        ]

    return penalties
