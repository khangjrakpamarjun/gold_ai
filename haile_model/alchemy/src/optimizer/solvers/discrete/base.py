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

"""Discrete base methods.
"""
from typing import Any, Callable, Dict, Hashable, List, Union

import numpy as np
import pandas as pd

from optimizer.domain import (
    CategoricalDimension,
    IntegerDimension,
    check_discrete_domain,
)
from optimizer.solvers import Solver


class DiscreteSolver(Solver):
    # pylint: disable=abstract-method
    def __init__(
        self,
        domain: List[Union[List[Any], IntegerDimension, CategoricalDimension]],
        sense: str = "minimize",
        seed: int = None,
    ):
        """Constructor.

        Args:
            domain: list describing the available choices for each dimension.
            sense: string, optional sense string for "minimize" or "maximize".
            seed: int, optional random seed.
        """
        super(DiscreteSolver, self).__init__(
            domain=check_discrete_domain(domain), sense=sense, seed=seed
        )

    # pylint: enable=abstract-method


class DomainMapper:
    """
    Small helper class to contain logic that maps elements of the domain to their
    indices in the domain's definition.
    """

    def __init__(
        self, domain_map: Dict[Hashable, int], strict: bool = False, default: int = None
    ):
        """Constructor.

        Args:
            domain_map: dictionary mapping domain values to integer indices. Should be
            the output of `make_domain_map`.
            strict: True to require that the values being mapped at inside `domain_map`.
            default: integer value to fill when strict is False. Has no effect if
            strict is True.
        """
        self.domain_map = domain_map

        if not strict and not isinstance(default, int):
            raise ValueError(
                f"Provided default value {default} of type "
                f"{type(default)} must be an integer."
            )

        self.strict = strict
        self.default = default

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Map the domain values in `x` to their integer indices

        Args:
            x: (m x n) numpy ndarray.

        Returns:
            (m x n) numpy array with only integer types.
        """
        out = np.empty(shape=x.shape, dtype=int)

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if self.strict:
                    mapped = self.domain_map[x[i, j]]

                else:
                    mapped = self.domain_map.get(x[i, j], self.default)

                out[i, j] = mapped

        return out


class IndexedDiscreteSolver(DiscreteSolver):
    # pylint: disable=abstract-method
    """
    Base class to handle the logic of converting indices of choices back to the choices
    themselves.

    For example, an index may be three possible strings ["a", "b", "c"] and a particular
    algorithm may only operate on integer values. The methods in this base class handle
    converting [0, 1, 2] back to ["a", "b", "c"] when outputting solutions. This way
    classes extending from here can deal purely with integer problems.
    """

    def __init__(
        self,
        domain: List[Union[List[Hashable], IntegerDimension, CategoricalDimension]],
        sense: str = "minimize",
        seed: int = None,
    ):
        """Constructor.

        Args:
            domain: list describing the available choices for each dimension.
            Note that the item here have to be hashable.
            sense: string, optional sense string for "minimize" or "maximize".
            seed: int, optional random seed.
        """
        super(IndexedDiscreteSolver, self).__init__(
            domain=domain, sense=sense, seed=seed
        )

        self.domain_lengths = np.array([len(d) for d in self._domain])
        self.domain_matrix = self.make_domain_matrix(self._domain)
        self.domain_map = self.make_domain_map(self._domain)
        self.domain_to_indices = self.wrap_domain_map(self.domain_map)

    @staticmethod
    def make_domain_matrix(domain: List[List[Hashable]]) -> np.ndarray:
        """Creates a matrix with potential values for each dimension.
        This is done to enable indexing with numpy interfaces.

        Indices with smaller domains will have their rows filled with the empty memory
        values for the datatype returned from np.empty.

        Examples:
            ::

                >>> IndexedDiscreteSolver.make_domain_matrix([["a", "b", "c"], [1, 2]])
                array([['a', 'b', 'c'], [1, 2, None]], dtype=object)

            The None value in the second row is really an undefined value. This index
            should (of course) not be used, so its actual value may differ based on the
            datatype of the row.

        Args:
            domain: list of lists describing the available choices for each dimension.

        Returns:
            A numpy ndarray, that in ith row contains the domain for the ith variable.
            If the variable has less discrete options than the dimension of the matrix,
            the row is filled with default values.
        """

        # Pandas handles typing for heterogeneous rows really well, so this is
        # the easiest way to retain numpy types where appropriate.
        return pd.DataFrame(domain).values

    @staticmethod
    def indices_to_domain(
        index_matrix: np.ndarray, domain_matrix: np.ndarray
    ) -> np.ndarray:
        """Convert the indices of domain values selected in `index_matrix` to actual
        domain values.

        Args:
            index_matrix: numpy matrix of integer indices.
            domain_matrix: numpy matrix of domain values. Should be the output of
            `make_domain_matrix`.

        Returns:
            np.array of the same dimension as `index_matrix` with domain values.
        """
        return domain_matrix[np.arange(domain_matrix.shape[0]), index_matrix]

    @staticmethod
    def make_domain_map(domain: List[List[Hashable]]) -> Dict[Hashable, int]:
        """Make a dictionary that maps an element of the domain to its index in
        the domain.

        Args:
            domain: list of lists describing the available choices for each dimension.

        Raises:
            TypeError: if an element of the domain is not hashable.

        Returns:
            dictionary.
        """
        domain_map = {}

        for dimension in domain:
            for index, domain_value in enumerate(dimension):
                if isinstance(domain_value, Hashable):
                    domain_map[domain_value] = index
                else:
                    raise TypeError(
                        f"Value {domain_value} of type {type(domain_value)} "
                        "in domain is not hashable."
                    )

        return domain_map

    @staticmethod
    def wrap_domain_map(
        domain_map: Dict[Hashable, int],
        strict: bool = True,
        default: int = 0,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Wrap the domain map dictionary output by `make_domain_map` with a Callable
        to handle the complexities of applying a dictionary to a numpy matrix.

        Args:
            domain_map: dictionary mapping domain values to integer indices. Should be
            the output of `make_domain_map`.
            strict: True to require that the values being mapped at inside `domain_map`.
            default: integer value to fill when strict is False. Has no effect if
            strict is True.

        Returns:
            Callable.

        Raises:
            ValueError: when default value is not none
        """
        return DomainMapper(domain_map, strict=strict, default=default)

    # pylint: enable=abstract-method
