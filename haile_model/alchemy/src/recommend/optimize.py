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
Core nodes performing the optimization.
"""
import logging
import operator
import typing as tp
from copy import deepcopy
from functools import partial
from typing import Dict, Optional

import pandas as pd
from joblib import Parallel, delayed

from optimizer.problem import OptimizationProblem
from optimizer.solvers import Solver
from optimizer.stoppers.base import BaseStopper

from .utils import get_penalty_slack

logger = logging.getLogger(__name__)


class SenseError(Exception):
    """Raise when problem and solver have different sense of optimization."""


def _safe_run(func):
    """
    Decorator for handling keyboard interupt in optimize function

    Args:
        func: function to decorate

    Returns:
        Empty tuple or results from func
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            logger.info(
                msg=f"Execution halted by user, raising KeyboardInterrupt."
                f"Was executing {args} and {kwargs}",
            )
            return ({}, {})

    return wrapper


@_safe_run
def optimize(
    problem: OptimizationProblem,
    solver: Solver,
    stopper: Optional[BaseStopper] = None,
) -> Solver:
    """
    Optimizes a single row and returns the ``Solver`` after doing the optimization.

    Args:
        problem (OptimizationProblem): optimization problem.
        solver (Solver): solver.
        stopper (Optional[BaseStopper]): stopper. Defaults to None.

    Raise:
        SenseError: if problem.sense != solver.sense

    Returns:
        Solver after optimization
    """
    if problem.sense != solver.sense:
        raise SenseError(
            "'problem' and 'solver' have different sense"
            f" of optimization: {problem.sense} != {solver.sense}",
        )
    stop_condition = False
    while not stop_condition:
        candidate_parameters = solver.ask()
        obj_vals, fixed_candidate_parameters = problem(candidate_parameters)
        solver.tell(fixed_candidate_parameters, obj_vals)
        stop_condition = solver.stop()
        if stopper is not None:
            stopper.update(solver)
            stop_condition |= stopper.stop()
    return solver


def bulk_optimize(
    problem_dict: Dict[tp.Any, OptimizationProblem],
    solver_dict: Dict[tp.Any, Solver],
    stopper: BaseStopper,
    n_jobs: float = 1,
) -> Dict[tp.Any, Solver]:
    """Run optimization in parallel.

    Args:
        problem_dict (Dict[pd.Index, OptimizationProblem]):
         dictionary containing problem.
        solver_dict (Dict[pd.Index, Solver]): dictionary containing `Solver`.
        stopper (BaseStopper): stopper.
        td (TagDict): tag dictionary.
        n_jobs: the number of cores used for parallel processing

    Raises:
        KeyError: if keys for `solver_dict` and `problem_dict` are different

    Returns:
        A dictionary with input data index and Solvers after optimization.
    """
    if set(solver_dict.keys()) != set(problem_dict.keys()):
        raise KeyError("'solver_dict' and 'problem_dict' must have same keys.")
    solvers_after_optimization = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(partial(optimize, stopper=stopper))(
            problem=problem_dict[idx],
            solver=solver_dict[idx],
        )
        for idx in problem_dict.keys()
    )
    return dict(zip(solver_dict.keys(), solvers_after_optimization))


def extract_optimization_result_from_solver(
    solver: Solver,
    problem: OptimizationProblem,
) -> pd.DataFrame:
    """
    Extract comparison of initial and optimized optimizable parameters.

    Args:
        solver: ``Solver`` instance that contains information about optimized rows
        problem: ``OptimizationProblem`` that was involved into optimization

    Returns:
        ``pd.DataFrame`` with comparison of initial and optimized rows
    """
    row = problem.state
    on_controls = problem.optimizable_columns

    # score with current and with optimal controls
    scores = pd.concat([row, row], ignore_index=True)
    scores.index = ["curr", "opt"]
    best_controls, _ = solver.best()
    scores.loc["opt", on_controls] = best_controls

    # Here we only evaluate the objective. This means penalties will not be applied.
    scores["objective"] = problem.objective(scores)
    penalty_slack_table = get_penalty_slack(scores, problem.penalties)
    scores = scores.join(penalty_slack_table, how="left")
    _check_invalid_solution(problem=problem, scores=scores)
    return scores


def extract_optimization_results_from_solvers(
    solver_dict: tp.Dict[tp.Any, Solver],
    problem_dict: tp.Dict[tp.Any, OptimizationProblem],
) -> tp.Dict[tp.Any, pd.DataFrame]:
    """
    Extracts comparison of initial and optimized
    optimizable parameters for each solver and problem
    using the ``extract_optimization_result_from_solver``.
    """
    return {
        problem_id: extract_optimization_result_from_solver(
            solver_dict[problem_id],
            problem_dict[problem_id],
        )
        for problem_id, problem in problem_dict.items()
    }


def _check_invalid_solution(problem: OptimizationProblem, scores: pd.DataFrame):
    """Replace invalid solutions to current state."""

    worse_comp = operator.gt if problem.sense == "minimize" else operator.lt
    opt_pred = scores.loc["opt", "objective"]
    curr_pred = scores.loc["curr", "objective"]

    # If we're strict, only return the optimized state if it's better than the current.
    if worse_comp(opt_pred, curr_pred):
        scores.loc["opt"] = deepcopy(scores.loc["curr"])
        problem_state_index = problem.state.index
        logger.warning(
            f"Optimized objective {opt_pred:0.4f} worse than current state objective "
            f"{curr_pred:0.4f} at index {problem_state_index}. Current state will be "
            "returned instead.",
        )
