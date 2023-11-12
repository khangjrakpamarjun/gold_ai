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
from .domain import BaseDomainGenerator, DiscreteDomainGenerator, MinMaxDomainGenerator
from .export import (
    prepare_predictions,
    prepare_recommendations,
    prepare_runs,
    prepare_states,
    prepare_tags,
)
from .optimization_explainer import (
    create_grid_for_optimizable_parameter,
    create_optimization_explainer_plot,
)
from .optimize import (
    bulk_optimize,
    extract_optimization_result_from_solver,
    extract_optimization_results_from_solvers,
    optimize,
)
from .postprocessing import SummaryTable, generate_run_id, prepare_for_uplift_plot
from .utils import (
    get_linear_search_space,
    get_on_features,
    get_penalty_slack,
    get_possible_values,
    get_set_repairs,
    make_problem,
    make_solver,
    make_stopper,
)

__version__ = "0.14.1"
