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
Problem set-up diagnostics functions
"""

from copy import copy
from numbers import Real
from operator import le, lt
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from optimizer.constraint import InequalityConstraint
from optimizer.problem.problem import OptimizationProblem
from optimizer.solvers.continuous.base import to_limits
from optimizer.types import Matrix, Vector


def evaluate_on_historic_data(
    data: pd.DataFrame, problem: OptimizationProblem, bounds: List[Tuple]
) -> pd.DataFrame:
    """
    Evaluates the problem statement on historical data. This is
    helpful to sanity check the constraints and problem set-up.

    Args:
        data: input data
        problem: Optimization Problem
        bounds: List of bounds for each variables in the input data. It is expected
        that the bounds are ordered tuples (i.e. (min, max) and is provided for
        each column in the input data.)

    Returns:
        A dataframe with input data along with the evaluated penalties,
        slack, bounds check and objective values. If the inequality constraint
        is violated, the slack is NaN.

    Raises:
        ValueError: if number of columns in data does not match
        the number of bounds

    """
    if len(data.columns) != len(bounds):
        raise ValueError("Bounds list length do not match the number of columns")

    return _evaluate_on_historic_data(data, problem, bounds)


def _evaluate_on_historic_data(
    data: pd.DataFrame, problem: OptimizationProblem, bounds: List[Tuple]
) -> pd.DataFrame:
    input_data = copy(data)

    # Bounds
    check_bounds = _check_bounds(bounds, input_data)

    # Penalties
    penalty_list = problem.penalties
    penalty_table = get_penalties_table(input_data, penalty_list)

    # Objective Value
    objective_values = _get_objective_value(input_data, problem)

    # Slack Value
    slack_table = get_slack_table(input_data, penalty_list)

    return (
        input_data.join(check_bounds, how="left")
        .join(penalty_table, how="left")
        .join(slack_table, how="left")
        .join(objective_values, how="left")
    )


def _get_objective_value(data: pd.DataFrame, problem_stmt: OptimizationProblem):
    """
    Computes the objective value with and without penalty.
    Here, we pass the inputs to the "objective" function of the problem class, so that
    the states are not replaced to the user provided values.
    Similarly, the "apply_penalty" function is used to account for both maximization
    and minimization use-case.
    """

    objective_value = pd.Series(
        problem_stmt.objective(data), name="objective_value", index=data.index
    )

    # Note that we are adding penalty separately to the objective value rather
    # than calling problem_stmt(data), so that the states are historical values
    # from the data and not those provided by user
    objective_value_with_penalty = pd.Series(
        objective_value + problem_stmt.apply_penalties(data),
        name="objective_value_with_penalty",
    )

    return pd.concat([objective_value, objective_value_with_penalty], axis=1)


def _check_bounds(bounds: List[Tuple], data: pd.DataFrame):
    """
    Checks if each column is within the specified bounds. It is expected
    that the bounds are ordered tuples (i.e. (min, max) and provided for each column
    in the dataset and in the same order as the dataset.)
    Returns a dataframe with boolean flags.
    """
    bounds_stack = pd.DataFrame(
        np.stack(bounds), columns=["min", "max"], index=data.columns
    )
    bounds_table = pd.DataFrame(
        np.stack(
            [
                (bounds_stack.loc[col, "min"] <= data[col])
                & (data[col] <= bounds_stack.loc[col, "max"])
                for col in data.columns
            ],
            axis=1,
        ),
        columns=["within_bounds_" + col_name for col_name in data.columns],
        index=data.index,
    )
    return bounds_table


def get_penalties_table(solutions: pd.DataFrame, penalties: List) -> pd.DataFrame:
    """
    Computes absolute values for all penalties on the given data.
    Returns a dataframe with calculated penalties.

    Args:
        solutions: Dataframe with state and controls
        penalties: List of penalty constraints

    Returns:
        Dataframe with penalties for each constraint

    """
    if not penalties:
        return pd.DataFrame()
    penalty_matrix = np.stack([p(solutions) for p in penalties], axis=1)
    penalty_check = pd.DataFrame(
        penalty_matrix,
        columns=[p.name for p in penalties],
        index=solutions.index
        if hasattr(solutions, "index")
        else np.arange(len(solutions)),
    )
    return penalty_check


def get_slack_table(solutions: pd.DataFrame, penalties: List) -> pd.DataFrame:
    """
    Compute slack (distance between function and boundary) for the
    InequalityConstraints in the penalties list.
    The value denotes the amount of slack left before the constraint is violated.

    If the penalties are not named, their slack values are returned as "slack_1",
        "slack_2", "slack_3" ...

    Args:
        solutions: Dataframe with state and controls
        penalties: List of penalty constraints

    Returns:
        Dataframe with slack for each inequality penalty constraint

    """
    if not penalties:
        return pd.DataFrame()

    inequality_penalties = [
        p for p in penalties if isinstance(p.constraint, InequalityConstraint)
    ]

    if not inequality_penalties:
        return pd.DataFrame()

    slack_matrix = np.stack(
        [p.constraint(solutions) for p in inequality_penalties], axis=1
    )
    slack_matrix = np.where(slack_matrix > 0, np.NaN, slack_matrix * -1)
    slack_table = pd.DataFrame(
        slack_matrix,
        columns=[
            p.name.replace("_penalty", "_slack") if p.name is not None else f"{i}_slack"
            for i, p in enumerate(inequality_penalties)
        ],
        index=solutions.index
        if hasattr(solutions, "index")
        else np.arange(len(solutions)),
    )
    return slack_table


NON_CONVEX_EPS = 1e-6


def non_convexity_test(  # pylint: disable=too-many-locals
    objective: Callable[[Matrix], Vector],
    domain: List[Tuple[Real, Real]],
    n_samples: int = 10000,
    n_points_on_line: int = 10,
    strict: bool = False,
    seed: int = None,
    return_hits: bool = False,
) -> Union[bool, Tuple[bool, int]]:
    """A simple test for non-convexity exploiting the definition of a convex function.
    Returns True if the provided function is non-convex.

    In summary, this function randomly samples `n_samples` pairs of points and draws a
    line segment between them. It then tests `n_points_on_line` values on the line using
    the definition of convexity. If the test

    f(t * x_1 + (1 - t) * x_2) <= t * f(x_1) + (1 - t) * f(x_2)

    fails for any of the samples, we can conclude `objective` is non-convex. The t
    values will be generated at even intervals on the line between x_1 and x_2.

    See here for more on convex functions:
        https://en.wikipedia.org/wiki/Convex_function

    ** Note: this test returning a False result does NOT prove convexity.
    ** Note: this function involves 3 calls to the objective function using matrices
    with dimension (`n_samples` * `n_points_on_line`, len(domain))
    ** Note: by definition, a function defined on an integer domain is not convex.
    Consider converting your discrete domain to a continuous one when using this
    function.
    ** Note: when using this test for an optimization problem, an objective may seem
    convex, but still be part of a non-convex optimization problem if the constraints
    are non-convex.

    Args:
        objective: callable, the function to test for non-convexity.
        domain: list of tuples, the boundaries of the objective function.
        n_samples: number of pairs of points to sample.
        n_points_on_line: number of points to test on the line segment formed between
            the two randomly sampled points.
        strict: boolean, False to allow for equality in the non-convexity test.
        seed: random seed.
        return_hits: boolean, True to return the number of times the test failed.

    Returns:
        A boolean describing the result of the test. Value is True if the `objective`
        is non-convex. If `return_hits` is true, the number of times the test failed
        will be returned.
    """
    rng = check_random_state(seed)
    comparison = lt if strict else le

    limits = to_limits(domain)

    line_segment_samples = np.linspace(0, 1, num=n_points_on_line, endpoint=False)[1:]

    # Sample from [0, 1) and convert to the desired domain.
    sample = rng.random_sample(size=(n_samples * 2, len(domain)))
    sample = limits[0] + (limits[1] - limits[0]) * sample

    sample_x1, sample_x2 = sample[:n_samples], sample[n_samples:]

    # In order to make the testing operation fast, we repeat each sample for each
    # point on the line we'd like to test.
    sample_x1 = np.repeat(sample_x1, len(line_segment_samples), axis=0)
    sample_x2 = np.repeat(sample_x2, len(line_segment_samples), axis=0)

    # Similar to the above operation, we tile the line segment t values to obtain a
    # different t for each repeated sample.
    tiled_segment_samples = np.tile(line_segment_samples, n_samples)

    lhs = objective(
        tiled_segment_samples.reshape(-1, 1) * sample_x1
        + (1 - tiled_segment_samples.reshape(-1, 1)) * sample_x2
    )
    rhs = tiled_segment_samples * objective(sample_x1) + (
        1 - tiled_segment_samples
    ) * objective(sample_x2)

    tests = comparison(lhs, rhs + (NON_CONVEX_EPS if not strict else 0))
    test_result = not all(tests)

    if return_hits:
        return test_result, np.sum(~tests).item()

    else:
        return test_result


def estimate_smoothness(  # pylint: disable=too-many-locals
    objective: Callable[[Matrix], Vector],
    domain: List[Tuple[Real, Real]],
    n_samples: Union[int, np.ndarray] = 10000,
    seed: int = None,
    return_list: bool = False,
) -> Union[float, Tuple[float, List[float]]]:
    """Estimate the smoothness (condition number) of a function.

    This implements an estimate of the relative condition number for a random sample of
    points from the domain.

    See here for more on the condition number for several variables:
        https://en.wikipedia.org/wiki/Condition_number#Several_variables

    For a simplified formula, also see here:
        https://math.stackexchange.com/q/736022

    Note: this estimate is meant to compare functions on similar ranges. Comparing
    functions with very different ranges may lead to incorrect conclusions.

    Args:
        objective: callable, function to test smoothness.
        domain: list of tuples, the boundaries of the objective function.
        n_samples: number of points to estimate the relative condition number or a
            numpy array of points to calculate condition numbers at.
        seed: random seed.
        return_list: boolean, True to return the list of all estimated condition numbers

    Returns:
        float, the mean relative condition number. If ``return_list`` is true, all
        estimates will also be returned.
    """
    rng = check_random_state(seed)
    limits = to_limits(domain)

    if isinstance(n_samples, int):
        samples = limits[0] + (limits[1] - limits[0]) * rng.rand(n_samples, len(domain))
    else:
        samples = np.array(n_samples, dtype=float)
        n_samples = samples.shape[0]

        if len(domain) != samples.shape[1]:
            raise ValueError(
                f"Given domain has {len(domain)} dimensions, but provided samples "
                f"have {samples.shape[1]} dimensions. Must be equal."
            )

    x_normed = np.linalg.norm(samples, ord=2, axis=1)
    fx_abs = np.abs(objective(samples))

    #
    # See here for an explanation on the arithmetic below in 1D: https://w.wiki/4Uuw
    # Here, matrices are constructed to vectorize the Jacobian estimate rather than
    # looping over the samples matrix and computing one at a time.
    #

    # Repeat the samples rowwise, once for each dimension.
    samples_repeated = np.repeat(samples, len(domain), axis=0)

    # Tile the identity matrix and multiply by the repeated samples to get only the
    # diagonals the of the repeated samples.
    tiled_eye = np.tile(np.eye(len(domain)), (n_samples, 1))

    # Compute the steps in each direction using an appropriate step size based on dtype.
    # See page 7 of:
    #   http://paulklein.ca/newsite/teaching/Notes_NumericalDifferentiation.pdf
    eps = np.power(np.finfo(samples.dtype).eps, 1 / 3)
    steps = (np.maximum(np.abs(samples_repeated), 1) * tiled_eye) * eps

    # Compute how far we stepped directly rather than reusing the above epsilon.
    forward_step = samples_repeated + steps
    backward_step = samples_repeated - steps
    dx = forward_step - backward_step

    # Get the diagonal of each stacked dx matrix.
    dx = dx[np.arange(dx.shape[0]), np.tile(np.arange(dx.shape[1]), n_samples)]

    # Estimate the Jacobian and take the L2 norm.
    jacobian_norm = np.linalg.norm(
        (  # Center point method for each partial derivative in the Jacobian.
            (objective(forward_step) - objective(backward_step)) / dx
        ).reshape(
            samples.shape
        ),  # Reshape to take the Jacobian across rows.
        ord=2,
        axis=1,
    )

    # Calculate the array of relative condition numbers.
    relative_condition_numbers = (x_normed * jacobian_norm) / fx_abs
    mean_condition = np.mean(relative_condition_numbers).item()

    if return_list:
        return mean_condition, list(relative_condition_numbers)

    else:
        return mean_condition
