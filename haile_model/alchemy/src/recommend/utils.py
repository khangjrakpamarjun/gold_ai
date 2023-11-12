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
Recommend pipeline utils.
"""
import importlib
import logging
import typing as tp
from copy import deepcopy

import numpy as np
import pandas as pd

from optimizer.constraint import Penalty, Repair, repair
from optimizer.problem import StatefulOptimizationProblem
from optimizer.solvers import Solver
from optimizer.solvers.discrete.base import DiscreteSolver
from optimizer.stoppers.base import BaseStopper
from optimizer.types import Matrix, Vector
from optimizer.utils.diagnostics import get_penalties_table, get_slack_table
from optimus_core.tag_dict import TagDict
from optimus_core.utils import env_var_to_bool

from .domain import BaseDomainGenerator

logger = logging.getLogger(__name__)

RepairList = tp.List[tp.Union[Repair, tp.Callable[[Matrix], Matrix]]]
PenaltyList = tp.List[tp.Union[Penalty, tp.Callable[[Matrix], Vector]]]


def load_obj(obj_path: str, default_obj_path: str = "") -> tp.Any:
    """Extract an object from a given path.

    Args:
        obj_path: Path to an object to be extracted, including the object name.
        default_obj_path: Default object path. In the case this is provided, `obj_path`
        must be a single name of the object being imported.

    Returns:
        Extracted object.

    Raises:
        AttributeError: When the object does not have the given named attribute.

    Examples:
        Importing an object::

            >>> load_obj("sklearn.linear_model.Ridge")

        Importing using `default_obj_path`::

            >>> load_obj("Ridge", default_obj_path="sklearn.linear_model")
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    loaded_object = getattr(module_obj, obj_name, None)
    if loaded_object is None:
        raise AttributeError(
            f"Object `{obj_name}` cannot be loaded from `{obj_path}`.",
        )
    return loaded_object


class DuplicateIndexError(Exception):
    """Raise when data contains duplicate index."""


def get_penalty_slack(solutions: pd.DataFrame, penalties: tp.List) -> pd.DataFrame:
    """This function generates penalties and slacks for all constraints.
    Slack is calculated only for inequality constraints

    Args:
        solutions (pd.DataFrame): optimization outputs.
        penalties (List): list of penalties.

    Returns:
        pd.DataFrame: tables with penalties and slacks.
    """
    if not penalties:
        return pd.DataFrame(index=solutions.index)

    penalty_table = get_penalties_table(solutions, penalties)
    slack_table = get_slack_table(solutions, penalties)
    return penalty_table.join(slack_table, how="left")


def _check_on_flag(row: pd.DataFrame, td: TagDict, feature: str) -> bool:
    """Check if any dependencies related to a feature is off.

    Args:
        row (pd.DataFrame): single optimization result
        td (TagDict): tag dictionary
        feature (str): name of feature to check dependency

    Returns:
        bool: True if all depencies are on.
    """

    dependencies = td.dependencies(feature)
    dependency_check = [row[dependency].iloc[0] > 0.5 for dependency in dependencies]

    return all(dependency_check)


def get_on_features(
    current_value: pd.DataFrame,
    td: TagDict,
    controls: tp.List[str],
) -> tp.List[str]:
    """Determine which features are "on" using the tag dictionary.

    Args:
        current_value: single row DataFrame describing the current values of features.
        td: tag dictionary.
        controls: controllable features used to determine on/off flag.

    Returns:
        List of strings, names of features that are "on".
    """
    return [
        feature for feature in controls if _check_on_flag(current_value, td, feature)
    ]


def make_solver(
    solver_class: tp.Type[Solver],
    solver_kwargs: tp.Dict[str, tp.Union[tp.List, str, float]],
    domain_dict: tp.Dict[pd.Index, tp.List[tuple]],
) -> tp.Dict[pd.Index, Solver]:
    """Creates solver for each index in `domain_dict`.

    Args:
        solver_class: Solver class from `optimizer` package.
        solver_kwargs: kwargs needed to construct solver_class besides domain.
        domain_dict: dict of domain bounds.

    Raises:
        NotImplementedError: when MixedDomainSolver or DiscreteSolver type is given.

    Returns:
        Dict of index from domain_dict and Solver object.
    """

    solver_dict = {}
    for idx, domain in domain_dict.items():
        solver_kwargs.update({"domain": domain})
        solver_dict[idx] = solver_class(**deepcopy(solver_kwargs))

    return solver_dict


def _is_step_size_set(td: TagDict, column: str) -> bool:
    step_size = td[column]["step_size"]
    return not np.isnan(step_size)


def get_set_repairs(td: TagDict, on_controls: tp.List[str]) -> tp.List[Repair]:
    """Generate list of `Repair` for each control variables in `on_controls`.

    Args:
        td (TagDict): tag dictionary
        on_controls (List[str]): list of control variables
         that can be optimized right now

    Returns:
        List[Repair]: list of `Repair` for control variables
    """
    return [
        _make_set_repair(td, col) for col in on_controls if _is_step_size_set(td, col)
    ]


def make_problem(  # noqa: WPS210,WPS231
    data: pd.DataFrame,
    td: TagDict,
    objective: tp.Callable[[pd.DataFrame], pd.Series],
    domain_generator: tp.Union[str, tp.Type[BaseDomainGenerator]],
    repairs: tp.Optional[tp.Union[Repair, RepairList]] = None,
    penalties: tp.Optional[tp.Union[Penalty, PenaltyList]] = None,
    sense: tp.Literal["minimize", "maximize"] = "maximize",
) -> tp.Tuple[tp.Dict, tp.Dict, tp.List]:
    """Generate a problem dictionary for optimizer.

    Args:
        data: input data.
        td: tag dictionary.
        objective: objective function.
        domain_generator: domain generator class or import path to that class.
        repairs: a repair function or a list of repairs. Defaults to None.
        penalties: a penalty function or a list of
         penalty functions. Defaults to None.
        sense: either "maximize" or "minimize". Defaults to "maximize".

    Raises:
        DuplicateIndex: if input data contains duplicated index.

    Returns:
        A tuple of problem_dict and bounds_dict.
        problem_dict is a mapping of data.index to StatefulOptimizationProblem.
        domain_dict is a mapping of data.index to domain space.
    """
    if isinstance(domain_generator, str):
        domain_generator = load_obj(domain_generator)
    controls = td.select("tag_type", "control")
    _check_control_config(td, controls)

    # do not use parallel processing in the model
    # since we are parallelizing over rows in the dataframe
    if getattr(objective, "estimator__n_jobs", None) is not None:
        objective.set_params(estimator__n_jobs=1)

    problem_dict = {}
    domain_dict = {}
    off_idx_list = []

    # make `repairs` into a list if single repair is given
    if isinstance(repairs, Repair) or callable(repairs):
        repairs = [repairs]

    repairs = repairs or []

    if np.any(data.index.duplicated()):
        raise DuplicateIndexError("'data' must have unique index.")

    for idx in data.index:
        row = data.loc[[idx]]
        on_controls = get_on_features(row, td, controls)

        if not on_controls:
            off_idx_list.append(idx)
            continue

        repairs = repairs + get_set_repairs(td, on_controls)

        domain_space = domain_generator(
            td=td,
            data=row,
            optimizables=on_controls,
        ).generate()

        problem = StatefulOptimizationProblem(
            objective,
            state=row,
            optimizable_columns=on_controls,
            repairs=repairs,
            penalties=penalties,
            sense=sense,
        )
        problem_dict[idx] = problem
        domain_dict[idx] = domain_space

    if off_idx_list:
        off_idx_count = len(off_idx_list)
        logger.info(
            f"{off_idx_count} rows are excluded since there is"
            " no active control variables.",
        )
    return problem_dict, domain_dict


def _make_set_repair(td: TagDict, column: str) -> Repair:
    """Creates a new constraint set repair for a given column"""

    op_max = float(td[column]["op_max"])
    op_min = float(td[column]["op_min"])
    step_size = float(td[column]["step_size"])

    constraint_set = get_linear_search_space(op_min, op_max, step_size)

    return repair(column, "in", constraint_set)


def get_linear_search_space(
    op_min: float,
    op_max: float,
    step_size: float,
) -> tp.List[float]:
    """This function creates linear search space using values from tag dictionary.

    Examples::

        >>> get_linear_search_space(2.0, 4, 0.5)
        ... np.array([2. , 2.5, 3.0 , 3.5 , 4.])

    Args:
        op_min (float): minimum value of linear search space
        op_max (float): maximum value of linear search space
        step_size (float): distance between each value in linear search space

    Returns:
        List[float]: linear search space
    """
    num_samples = np.round((op_max - op_min) / step_size, 0).astype("int")
    lin_space = np.linspace(start=op_min, stop=op_max, num=num_samples + 1)

    return list(lin_space)


def _check_control_config(td: TagDict, controls: tp.List[str]):
    """Checking if configs on control variables are missing from tag dict"""

    td_df = td.to_frame()
    conf_to_check = ["op_min", "op_max"]
    null_check = td_df.loc[td_df["tag"].isin(controls)][conf_to_check].isnull().any()

    if null_check.any():
        null_tags = null_check.loc[null_check].index.values
        raise KeyError(
            f"{conf_to_check} is missing from control variables: {null_tags}",
        )


def make_stopper(
    stopper_class: BaseStopper,
    stopper_kwargs: tp.Dict[str, tp.Dict],
) -> BaseStopper:
    return stopper_class(**stopper_kwargs)


def get_possible_values(solver: Solver, on_controls: tp.List[str]) -> tp.Dict[str, str]:
    not_discrete_solver = not isinstance(solver, DiscreteSolver)
    if not_discrete_solver or not env_var_to_bool("DYNAMIC_RECS_ENABLED"):
        return {control: [] for control in on_controls}
    return dict(zip(on_controls, solver._domain))  # noqa: WPS437
