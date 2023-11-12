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
This is a boilerplate pipeline
"""

from .feature_factory import FeatureFactory, create_features, draw_graph
from .pydantic_models import DerivedFeaturesCookBook, DerivedFeaturesRecipe
from .sample_function import (
    pandas_divide,
    pandas_max,
    pandas_mean,
    pandas_min,
    pandas_prod,
    pandas_subtract,
    pandas_sum,
)

__version__ = "0.8.4"
