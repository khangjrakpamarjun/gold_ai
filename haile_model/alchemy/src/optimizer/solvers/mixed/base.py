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

"""MixedDomainSolver base class definition.
"""

import numbers
from typing import Any, List, Tuple, Union

from optimizer.domain import Domain
from optimizer.domain.base import BaseDimension
from optimizer.solvers.base import Solver


class MixedDomainSolver(Solver):
    # pylint: disable=abstract-method

    def __init__(
        self,
        domain: Union[
            Domain,
            List[
                Union[
                    Tuple[numbers.Real, numbers.Real],
                    Tuple[numbers.Real, numbers.Real, str],
                    List[Any],
                    BaseDimension,
                ]
            ],
        ],
        sense: str = "minimize",
        seed: int = None,
    ):
        """Constructor.

        Args:
            domain: list of describing the domain along each dimension or Domain object.
            sense: string, optional sense string for "minimize" or "maximize".
            seed: int, optional random seed.
        """
        super().__init__(domain=domain, sense=sense, seed=seed)

        if not isinstance(domain, Domain):
            self._domain = Domain(domain)

    # pylint: enable=abstract-method
