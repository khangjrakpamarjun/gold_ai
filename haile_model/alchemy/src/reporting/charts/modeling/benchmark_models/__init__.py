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
Contains simple benchmark models that are used as a reference in the current model
 performance analyses
"""

# TODO: Find a solution to the noqas when importing protected methods
#  either import them right, or make them public or something
#  There must be a better way to do it!
#  We should find a way to solve these ignored WPS450A!
from ._autoregressive import AutoregressiveBenchmarkModel
from ._benchmark_model_base import BenchmarkModelBase
from ._defaults import fit_default_benchmark_models
from ._moving_average import MovingAverageBenchmarkModel
from ._regression_metrics import evaluate_regression_metrics
