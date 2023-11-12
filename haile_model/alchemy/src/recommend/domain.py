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

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Container, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from optimus_core import TagDict

logger = logging.getLogger(__name__)

TRange = Tuple[float, float]

_DISCRETE_DOMAIN_STEP_NUM = 12


class BaseDomainGenerator(ABC):
    """Abstract class for generating domain space."""

    def __init__(
        self,
        td: TagDict,
        data: pd.DataFrame,
        optimizables: List[str],
    ) -> None:
        """Construct BaseDomainGenerator class.

        Args:
            td: tag dictionary
            data: input data to generate domain space
            optimizables: a list of optimizable variables in input data
        """
        self._td = td
        self._data = data
        self._optimizables = optimizables
        self.is_optimizables_empty = self._is_optimizables_empty()

    @property
    def td(self) -> TagDict:
        """
        A property containing tag dictionary.

        Returns:
            Tag dictionary
        """
        return deepcopy(self._td)

    @property
    def data(self) -> pd.DataFrame:
        """
        A property containing input data.

        Returns:
            Input data
        """
        return deepcopy(self._data)

    @property
    def optimizables(self) -> List[str]:
        """
        A property containing the list of optimizable variables.

        Returns:
            List of optimizable variables
        """
        return deepcopy(self._optimizables)

    @abstractmethod
    def generate(self) -> List[Container[float]]:
        """Generate domain space."""

    def __repr__(self) -> str:
        """Generate string representation of the class."""
        return str(self.__class__.__name__)

    def _is_optimizables_empty(self):
        """Check if 'optimizables' is empty."""
        if self.optimizables:
            return False
        logger.warning("There's no variables to optimize.")
        return True


class MinMaxDomainGenerator(BaseDomainGenerator):
    """A MinMax domain generator for continuous solvers."""

    def generate(self) -> List[Tuple[float, float]]:
        """Generate min/max domain space for continuous solvers.

        If optimizable variables are missing, then return empty domain space

        Returns:
            List of tuples containing lower and upper bounds for optimizable variables
        """
        domain_space = []
        for optimizable in self.optimizables:
            current_value = _get_current_value(data=self.data, optimizable=optimizable)
            op_min, op_max, max_delta = _get_single_optimizable_domain_attributes(
                td=self.td,
                optimizable=optimizable,
            )
            lower_bound, upper_bound = _determine_single_variable_optimization_range(
                curr_value=current_value,
                op_min=op_min,
                op_max=op_max,
                max_delta=max_delta,
            )
            domain_space.append((lower_bound, upper_bound))
        return domain_space


class DiscreteDomainGenerator(BaseDomainGenerator):
    """A domain generator for discrete solvers."""

    def generate(self) -> List[List[float]]:
        """Generate discrete domain space for discrete solvers.

        If optimizables are missing, then return empty domain space.

        Raises:
            ValueError: raised if ``step_size`` is missing.

        Returns:
            List of linear space for each optimizable variable.
        """
        domain_space = []
        for optimizable in self.optimizables:
            discrete_domain = _calculate_discrete_domain(
                data=self.data,
                td=self.td,
                optimizable=optimizable,
            )
            domain_space.append(discrete_domain)
        return domain_space


def _calculate_discrete_domain(
    data: pd.DataFrame,
    td: TagDict,
    optimizable: str,
) -> np.ndarray:
    """Calculate discrete domain for a single variable."""
    current_value = _get_current_value(data=data, optimizable=optimizable)
    op_min, op_max, max_delta = _get_single_optimizable_domain_attributes(
        td=td,
        optimizable=optimizable,
    )
    lower_bound, upper_bound = _determine_single_variable_optimization_range(
        curr_value=current_value,
        op_min=op_min,
        op_max=op_max,
        max_delta=max_delta,
    )
    step_size = float(td[optimizable]["step_size"])
    discrete_domain = _generate_discrete_domain_for_single_variable(
        op_min=op_min,
        op_max=op_max,
        step_size=step_size,
    )
    return _restrict_discrete_domain_to_within_bounds(
        discrete_domain=discrete_domain,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


def _restrict_discrete_domain_to_within_bounds(
    discrete_domain: Iterable[float],
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    """Keep only values between ``lower_bound`` and ``upper_bound``.

    Args:
        linear_space: linear space
        lower_bound: lower bound
        upper_bound: upper bound

    Returns:
        Updated linear space.
    """
    is_within_bounds = np.logical_and(
        discrete_domain >= lower_bound,
        discrete_domain <= upper_bound,
    )
    return discrete_domain[is_within_bounds]


def _determine_single_variable_optimization_range(
    curr_value: float,
    op_min: float,
    op_max: float,
    max_delta: Optional[float],
) -> TRange:
    """
    Determines the optimization range based on max step and operations min and max.

    Determines the optimization range for a single variable (intended for domains where
    the domains of the variables are defined independent of one another, can be applied
    in some cases where the domain for some of the variables can be defined in this
    way).

    Solves edge cases in the following way:
    - if ``curr_value`` is between ``op_min`` and ``op_max``, returns the intersection
      of [``op_min``, ``op_max``] and
      [``curr_value`` - ``max_delta``, ``curr_value``+``max_delta``]
    - if ``curr_value`` is below ``op_min``, returns
      [``op_min``, ``op_min``+``max_delta``]
    - if ``curr_value`` is above ``op_max``, returns
      [``op_min`` - ``max_delta``, ``op_max``]
    """
    if curr_value > op_max:
        logger.warning(
            "Current value is above `op_max`."
            " Proceeding as if the current value was equal to `op_max`",
        )
        curr_value = op_max
    if curr_value < op_min:
        logger.warning(
            "Current value below `op_min`"
            " Proceeding as if the current value was equal to `op_min`",
        )
        curr_value = op_min
    if (max_delta is None) or pd.isna(max_delta):
        return op_min, op_max
    lower_bound = max(op_min, curr_value - max_delta)
    upper_bound = min(op_max, curr_value + max_delta)
    return lower_bound, upper_bound


def _get_current_value(data: pd.DataFrame, optimizable: str) -> float:
    """Get the current value of the optimizable variable from the data."""
    return data[optimizable].iloc[0]


def _get_single_optimizable_domain_attributes(
    td: TagDict,
    optimizable: str,
) -> Tuple[float, float]:
    """Get attributes relevant for single-optimizable domain the tag dict.

    Notes:
        These are common attributes for building a domain where the optimizable
        variable is meant to be within a given interval, independently of the values of
        the other variables in the optimization problem.

        These attributes are:
        - op_max
        - op_min
        - max_delta

    Args:
        td: tag dictionary
        optimizable: name of optimizable variable

    Raises:
        ValueError: if ``op_min`` or ``op_max`` is missing
        ValueError: if ``op_min`` is greater than ``op_max``

    Returns:
        (op_min, op_max)
    """

    optimizable_attrs = td[optimizable]

    op_min = optimizable_attrs["op_min"]
    if pd.isna(op_min):
        raise ValueError("'op_min' is required to generate domain space.")
    op_max = optimizable_attrs["op_max"]
    if pd.isna(op_max):
        raise ValueError("'op_max' is required to generate domain space.")
    if op_max < op_min:
        raise ValueError("'op_max' must be greater than equal to 'op_min'.")
    max_delta = optimizable_attrs["max_delta"]
    return op_min, op_max, max_delta


def _generate_discrete_domain_for_single_variable(
    op_min: float,
    op_max: float,
    step_size: Optional[float],
) -> np.ndarray:
    """Generate a discretized domain.

    Notes:
        - If ``step_size`` is not provided, creates an evenly spaced domain of
          ``_DISCRETE_DOMAIN_STEP_NUM`` points between ``op_min`` and ``op_max``.
        - If ``step_size`` is provided, creates an evenly spaced domain starting from
          ``op_min`` and adding points with step equal to ``step_size``.
          Ensures that ``op_max`` is present in the domain by adding it to the array
          if it is not present (this happens if ``op_max`` - ``op_min`` is not a
          multiple of ``step_size``).
    """
    if (step_size is None) or pd.isna(step_size):
        lin_space, step_size = np.linspace(
            start=op_min,
            stop=op_max,
            num=_DISCRETE_DOMAIN_STEP_NUM,
            retstep=True,
        )
        logger.info(
            f"`step_size` not provided."
            f" Divided the operational range into steps of size {step_size}"
            f" This domain consists of {_DISCRETE_DOMAIN_STEP_NUM} points",
        )
        return lin_space
    lin_space = np.arange(start=op_min, stop=op_max, step=step_size)
    if op_max not in lin_space:
        lin_space = np.append(lin_space, op_max)
    num_points = len(lin_space)
    logger.info(
        f"Created an evenly spaced domain starting from {op_min} with step {step_size}"
        f" This domain consists of {num_points} points",
    )
    return lin_space
