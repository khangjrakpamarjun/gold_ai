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
Holds definitions for the OptimizationProblem.
"""

from typing import Callable, List, Tuple, Union

import numpy as np

from optimizer.constraint.penalty import Penalty
from optimizer.constraint.repair import Repair
from optimizer.exceptions import InvalidObjectiveError
from optimizer.types import Matrix, Vector
from optimizer.utils import check_matrix


class OptimizationProblem:
    """
    Represents a basic optimization problem.
    Meaning all variables passed to __call__ are optimizable.
    """

    def __init__(
        self,
        objective: Callable[[Matrix], Vector],
        penalties: Union[
            Penalty,
            Callable[[Matrix], Vector],
            List[Union[Penalty, Callable[[Matrix], Vector]]],
        ] = None,
        repairs: Union[
            Repair,
            Callable[[Matrix], Matrix],
            List[Union[Repair, Callable[[Matrix], Matrix]]],
        ] = None,
        sense: str = "minimize",
    ):
        """Constructor.

        Args:
            objective: callable object representing the objective function.
            penalties: optional Penalties or callable or list of
                Penalties and/or callables.
            repairs: optional Repairs or callable or list of Repairs and/or callables.
                - Repairs are always applied before the objective and penalties are
                calculated.
            sense: 'minimize' or 'maximize', how to optimize the objective.

        Raises:
            InvalidObjectiveError: when objective is not callable or doesn't have a
                callable predict method.
            ValueError: if sense is not "minimize" or "maximize".
            ValueError: if penalties or repairs are invalid types/not callable.

        """
        if not callable(objective):
            if not (hasattr(objective, "predict") and callable(objective.predict)):
                raise InvalidObjectiveError(
                    "Provided objective must be callable or have a .predict method."
                )
        self._objective = objective

        if penalties is not None and (
            isinstance(penalties, Penalty) or callable(penalties)
        ):
            penalties = [penalties]
        self.penalties = penalties or []
        self._check_callable(self.penalties, "penalty")

        if repairs is not None and (isinstance(repairs, Repair) or callable(repairs)):
            repairs = [repairs]
        self.repairs = repairs or []
        self._check_callable(self.repairs, "repair")

        if sense not in ("maximize", "minimize"):
            raise ValueError(f"{sense} is an invalid optimization sense.")

        self.sense = sense

    def _check_callable(self, callables: List, required_type_str: str):
        for callable_obj in callables:
            if not callable(callable_obj):
                raise ValueError(
                    f"{type(callable_obj).__name__} provided as a {required_type_str}, "
                    f"but a {required_type_str} or valid callable is required."
                )

    @property
    def objective(self) -> Callable[[Matrix], Vector]:
        """returns callable objective"""
        if callable(self._objective):
            return self._objective
        return self._objective.predict

    def __call__(
        self,
        parameters: Matrix,
        skip_penalties: bool = False,
        skip_repairs: bool = False,
    ) -> Tuple[Vector, Matrix]:
        """Evaluate the OptimizationProblem on a Matrix of parameters.

        Args:
            parameters: Matrix of parameters to evaluate.
            skip_penalties: if True, evaluate without applying penalties.
            skip_repairs: if True, evaluate without applying repairs.

        Returns:
            Vector of objective values and (possibly repaired) parameter matrix.
        """
        check_matrix(parameters)

        if self.repairs and not skip_repairs:
            parameters = self.apply_repairs(parameters)
        objectives = self.objective(parameters)

        if self.penalties and not skip_penalties:
            objectives += self.apply_penalties(parameters)

        return objectives, parameters

    def apply_penalties(self, parameters: Matrix) -> Vector:
        """Apply penalties and calculate penalty.

        Args:
            parameters: Matrix of parameters for calculating penalties.

        Returns:
            Vector of penalty values.
        """
        total_penalty = np.zeros(parameters.shape[0])

        for penalty in self.penalties:
            total_penalty += penalty(parameters)

        if self.sense == "maximize":
            total_penalty *= -1

        return total_penalty

    def apply_repairs(self, parameters: Matrix) -> Vector:
        """Apply repairs.

        Args:
            parameters: Matrix of parameters to repair.

        Returns:
            Matrix.
        """
        repaired = parameters

        for repair in self.repairs:
            repaired = repair(repaired)

        return repaired
