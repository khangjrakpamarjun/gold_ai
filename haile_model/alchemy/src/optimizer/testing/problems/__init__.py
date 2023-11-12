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
Test problem package. Implements a selection of the CEC2005 test problems and \
elsewhere.
"""
from .ackley import Ackley  # noqa: F401
from .griewank import Griewank  # noqa: F401
from .max import Max  # noqa: F401
from .rastrigin import Rastrigin  # noqa: F401
from .sphere import Sphere  # noqa: F401
