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
N-Best logger module.
"""

import heapq as hq
from copy import deepcopy
from typing import Tuple

import numpy as np

from optimizer.loggers.base import LoggerMixin
from optimizer.solvers import Solver


class NBestLogger(LoggerMixin):
    """
    Logs the N best solutions over time.
    """

    def __init__(self, n: int, sense: str = "minimize"):
        """Constructor.

        Args:
            n: number of solutions to keep track of.
            sense: str, "minimize" or "maximize".
        """
        if n <= 0:
            raise ValueError(f"n must be greater than zero, got {n}.")

        if sense not in ["maximize", "minimize"]:
            raise ValueError(f'sense must be "minimize" or "maximize", got {sense}.')

        self.n = n
        self.heap = None
        self.sense = sense

    # pylint: disable=arguments-differ

    def log(self, solver: Solver):
        """Update the current best values.

        Args:
            solver: Solver object.
        """
        parameters, objectives = solver.parameters, solver.objective_values

        sign = 1 if self.sense == "minimize" else -1
        solutions = [(sign * o.item(), p) for o, p in zip(objectives, parameters)]

        hq.heapify(solutions)

        if self.heap is None:
            # We don't have a heap yet, so just get the smallest solutions.
            self.heap = hq.nsmallest(self.n, solutions)

        else:
            new_heap = []

            # Loop while we have enough solutions to push on the heap and the new
            # heap is less than the desired length.
            while (solutions or self.heap) and len(new_heap) < self.n:
                if not self.heap:  # Stored heap is empty.
                    hq.heappush(new_heap, hq.heappop(solutions))

                elif not solutions:  # Candidate solutions are empty.
                    hq.heappush(new_heap, hq.heappop(self.heap))

                elif solutions[0][0] < self.heap[0][0]:
                    hq.heappush(new_heap, hq.heappop(solutions))

                else:
                    hq.heappush(new_heap, hq.heappop(self.heap))

            self.heap = new_heap

    # pylint: enable=arguments-differ

    def n_best(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the N best logged solutions in sorted order and their objective values.

        If M < N values are logged throughout the lifetime of this object, this function
        will return M total values rather than N.

        Returns:
            Tuple of arrays. The first has shape (N, d), the second (N,).
            Where d is the dimension of the problem being solved.
        """
        if self.heap is None:
            raise RuntimeError("NBestLogger has not logged any values yet.")

        heap = deepcopy(self.heap)

        parameters, objectives = [], []

        sign = 1 if self.sense == "minimize" else -1
        while heap:
            solution = hq.heappop(heap)

            parameters.append(solution[1])
            objectives.append(sign * solution[0])

        return np.array(parameters), np.array(objectives)
