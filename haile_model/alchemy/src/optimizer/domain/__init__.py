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
Domain specification module.
"""

from optimizer.domain.categorical import CategoricalDimension  # noqa: F401
from optimizer.domain.domain import Domain  # noqa: F401
from optimizer.domain.integer import IntegerDimension  # noqa: F401
from optimizer.domain.real import RealDimension  # noqa: F401
from optimizer.domain.utils import (  # noqa: F401
    check_continuous_domain,
    check_discrete_domain,
)
