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

"""Grid search solver definition.
"""
import abc
import warnings
from itertools import product
from typing import Hashable, List, Tuple, Union

import numpy as np

from optimizer.domain import CategoricalDimension, IntegerDimension
from optimizer.solvers.discrete.base import IndexedDiscreteSolver
from optimizer.types import Matrix, Vector


class GridIterator(abc.ABC):
    """Base class for grid iterator."""

    def __init__(self, indices: List[List[int]], batch_size: int = None):
        """Constructor.

        Args:
            indices: list of lists describing the domain index choices.
            batch_size: integer number of points to return each call to `next`.
        """
        self.indices = indices
        self.batch_size = batch_size
        self.points_returned = 0
        self.grid = None

    def __len__(self) -> int:
        """Returns:
        int, the number of points on the grid.
        """
        return GridSearchSolver.n_grid_points(self.indices)

    def returned_all_points(self) -> bool:
        """Returns:
        True if all the points on the grid have been returned.
        """
        return self.points_returned >= len(self)

    @abc.abstractmethod
    def __next__(self) -> np.ndarray:
        """Returns:
        np.ndarray, one or more points on the grid.
        """

    def __iter__(self) -> "GridIterator":
        """Returns:
        GridIterator.
        """
        return self


class BatchGridIterator(GridIterator):
    """Batch grid iterator used in `GridSearchSolver`."""

    def __next__(self) -> np.ndarray:
        """Return the next `batch_size` points in the grid.

        If self.batch_size is None, return the whole grid.

        Raises:
            StopIteration: if we have returned all points on the grid.

        Returns:
            np.ndarray.
        """
        if self.returned_all_points():
            raise StopIteration

        if self.grid is None:
            self.grid = np.stack(np.meshgrid(*self.indices), -1).reshape(
                -1, len(self.indices)
            )

        batch = self.batch_size or len(self)
        out = self.grid[self.points_returned : self.points_returned + batch, :]
        self.points_returned += len(out)

        return out


class LowMemoryGridIterator(GridIterator):
    """Low memory grid iterator used in `GridSearchSolver`."""

    def __next__(self) -> np.ndarray:
        """Return `batch_size` points on the grid.

        Uses itertools.product to loop through the grid.

        Raises:
            StopIteration: if we have returned all points on the grid.

        Returns:
            np.ndarray.
        """
        if self.returned_all_points():
            raise StopIteration

        if self.grid is None:
            self.grid = product(*self.indices)

        out = []
        n = 0
        grid_size = len(self)
        batch = self.batch_size or len(self)
        while n < batch and n + self.points_returned < grid_size:
            out.append(next(self.grid))
            n += 1

        self.points_returned += len(out)

        return np.array(out)


class GridSearchSolver(IndexedDiscreteSolver):
    """Solver that simply returns all possible points on the grid formed by the domain."""

    def __init__(
        self,
        domain: List[Union[List[Hashable], IntegerDimension, CategoricalDimension]],
        sense: str = "minimize",
        seed: int = None,
        mode: str = "full",
        batch_size: int = None,
    ):
        """Constructor.

        Args:
            domain: list describing the available choices for each dimension.
            Note that the item here have to be hashable.
            sense: string, optional sense string for "minimize" or "maximize".
            seed: int, optional random seed.
            mode: str, the method for generating the grid. Options:
            - "full": generate and return the entire grid on the first call to `ask`.
            - "batch": generate and return only `batch_size` points on the grid on each
            call to `ask`.
            - "low_memory": generate and hold only `batch_size` points on the grid in
            memory at one time. Low memory is _much_ slower than "full" or "batch".
            batch_size: integer number of points on the grid to return at once. Has no
            effect if `mode = full`. Providing a batch size larger than the domain will
            return the entire domain at once (same effect as None).

        Raises:
            ValueError: if using `mode = full` or `batch` mode with a grid that cannot
            be stored in a numpy array (would take up more bits than the max `np.intp`).

        Warnings:
            RuntimeWarning: when `mode = low_memory` indicating execution will be slow.
            RuntimeWarning: when `batch_size != None` and `mode == batch`.
        """
        n_points = self.n_grid_points(domain)

        if mode in ["full", "batch"] and not self.grid_fits_in_numpy_array(n_points):
            raise ValueError(
                f"Provided domain would generate a grid with {n_points} points which "
                f"cannot be fit in a numpy array. Change `mode` to 'low_memory' or "
                "reduce the granularity or dimensionality of your domain."
            )

        if mode in ["full", "batch"] and len(domain) > 32:
            raise ValueError(
                f'Cannot use grid search mode "{mode}" with a '
                f"{len(domain)}-dimensional domain. Maximum is 32."
            )

        if mode not in ["full", "batch", "low_memory"]:
            raise ValueError(f"Invalid grid search mode {mode}.")

        if mode == "low_memory":
            warnings.warn(
                "Low memory mode will cause grid search to run slowly. "
                "Consider reducing the granularity of your domain and using "
                "full or batch mode.",
                category=RuntimeWarning,
            )

        if mode == "full" and batch_size is not None:
            warnings.warn(
                "Batch size has no effect when `mode = full`. Change mode to `batch` if"
                f"batches of size {batch_size} are desired.",
                category=RuntimeWarning,
            )

        super(GridSearchSolver, self).__init__(domain, sense=sense, seed=seed)

        if mode == "full":
            batch_size = None

        self.mode = mode
        self.batch_size = batch_size

        indices = [list(range(length)) for length in self.domain_lengths]
        self.grid_iterator = (
            BatchGridIterator(indices, batch_size=batch_size)
            if mode in ["full", "batch"]
            else LowMemoryGridIterator(indices, batch_size=batch_size)
        )

        self.current_grid_points = None
        self.current_objective_values = None

        self.best_point = None
        self.best_objective = float("inf")

    @staticmethod
    def n_grid_points(domain: List[List[Hashable]]) -> int:
        """Determine how many grid points are on the provided domain.

        Args:
            domain: list of lists describing the available choices for each dimension.

        Returns:
            int.
        """
        out = None

        if domain:
            for dimension in domain:
                n = len(dimension)

                if n > 0 and out is None:
                    out = n
                elif n > 0:
                    out *= n

        return out or 0

    @staticmethod
    def grid_fits_in_numpy_array(grid_size: int) -> bool:
        """Determine if a grid of the given size will fit in a numpy array.

        We don't check if the values in the domain themselves will fit because we will
        only need to store the integer indices of the domain values.

        Args:
            grid_size: integer size of the grid.

        Returns:
            bool, True if the grid will fit into a numpy array as integers indices.
        """
        return (
            grid_size * np.intp().nbytes < np.iinfo(np.intp).max
        )  # pylint: disable=E1120

    def ask(self) -> np.ndarray:
        """Get a collection of points in the defined grid.

        How many points will depend on `self.batch_size`. The order and method of point
        generation is defined by `self.mode`.

        Returns:
            np.ndarray.
        """
        return self.indices_to_domain(next(self.grid_iterator), self.domain_matrix)

    def tell(self, parameters: Matrix, objective_values: Vector):
        """Set the parameter and objective values.

        If necessary, update the current best point and objective value.

        Args:
            parameters: Matrix of parameter values.
            objective_values: Vector of objectives.
        """
        super(GridSearchSolver, self).tell(parameters, objective_values)
        parameters, objective_values = self._process_told_parameters_and_objectives(
            parameters, objective_values
        )

        self.told = True

        best_idx = np.argmin(objective_values)

        if objective_values[best_idx] < self.best_objective:
            self.best_point = parameters[best_idx]
            self.best_objective = objective_values[best_idx]

        self.current_grid_points = parameters
        self.current_objective_values = objective_values

    def stop(self) -> bool:
        """Determine if we should stop iterating the Ask and Tell loop.

        Returns:
            Boolean, True if we should stop.
        """
        return self.grid_iterator.returned_all_points()

    def best(self) -> Tuple[np.ndarray, float]:
        """Get the best point and objective found by grid search.

        Returns:
            Vector and float, the point and its objective value.
        """
        if self.best_point is None or self.best_objective is None:
            raise RuntimeError(
                "No parameters or objectives have been told to the solver yet."
            )

        return self.best_point, self.best_objective

    @property
    def _internal_objective_values(self) -> Vector:
        """Get internal objective values.

        Returns:
            Vector of the current objective values.
        """
        return self.current_objective_values

    @property
    def parameters(self) -> Matrix:
        """Get the most recently told grid points.

        Returns:
            Matrix of current parameters.
        """
        return self.current_grid_points
