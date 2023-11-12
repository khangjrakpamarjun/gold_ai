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
Maximum objective based stopper.
"""

from numbers import Real

import numpy as np

from optimizer.solvers.base import Solver
from optimizer.stoppers.base import BaseStopper
from optimizer.stoppers.utils import get_best_maximize, get_best_minimize


class NoImprovementStopper(BaseStopper):
    """
    This class will stop the search after N iterations without improvement in
    the best objective value.
    """

    def __init__(
        self, patience: int, sense: str, min_delta: Real = 0.0, first_delta: Real = 0.0
    ):
        """Constructor.

        Args:
            patience: number of iterations without improvement.
            sense: "minimize" or "maximize".
            min_delta: minimum difference to be considered an improvement.

                - Zero means any improvement will be reset the patience counter.
                - Must be positive, any improvement will be compared in the appropriate
                  direction whether we are maximizing or minimizing.

            first_delta: minimum difference between the first best solution to be
                considered an improvement.

                - Counting up toward `patience` will not begin until a solution that
                  is at least `first_delta` better than the first best solution.

        Raises:
            ValueError: if sense is not "minimize" or "maximize".
            ValueError: if min_delta is negative.
            ValueError: if first_delta is negative.
        """
        self.no_improvement = 0
        self.patience = patience

        if sense == "minimize":
            self.get_best = get_best_minimize

        elif sense == "maximize":
            self.get_best = get_best_maximize

        else:
            raise ValueError(f"Invalid sense {sense} provided.")

        self.best = np.PINF if sense == "minimize" else np.NINF

        if min_delta < 0:
            raise ValueError(f"min_delta must be positive. Provided {min_delta}.")

        if first_delta < 0:
            raise ValueError(f"first_delta must be positive. Provided {first_delta}.")

        self.min_delta = min_delta
        self.first_delta = first_delta
        self.first_best = None

    def _update(self, best: Real, delta: float):
        """Internal update. Updates counters based on improvements.

        Args:
            delta: float, delta between external best and internal best solution.
        """
        if delta > self.min_delta:
            # We found a better objective value.
            self.best = best
            self.no_improvement = 0

        else:
            self.no_improvement += 1

    def update(self, solver: Solver, **kwargs):  # pylint: disable=arguments-differ
        """Update internal state based on best objective value seen so far.

        Args:
            solver: Solver object.
            **kwargs: unused keyword arguments.

        Returns:
            bool, True if the search should stop.
        """
        best, delta = self.get_best(solver.objective_values, self.best)

        if self.first_delta > 0.0:
            self.first_best = best if self.first_best is None else self.first_best
            _, delta_btw_first = self.get_best([best], self.first_best)

            # Only begin normal operation after we've found something better than the
            # first best solution by at least `self.first_delta`.
            if self.first_best != best and delta_btw_first > self.first_delta:
                self._update(best, delta)

        else:
            self._update(best, delta)

        self._stop = self.no_improvement >= self.patience
