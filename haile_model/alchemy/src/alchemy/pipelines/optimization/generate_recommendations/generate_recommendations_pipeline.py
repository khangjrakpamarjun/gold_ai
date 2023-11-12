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


from kedro.pipeline import Pipeline, node

from alchemy.pipelines.optimization.optimization.objective import make_objective
from alchemy.pipelines.optimization.optimization.optimization_nodes import (
    add_model_predictions,
    adjust_bounds_dict,
    make_problem,
)
from alchemy.pipelines.optimization.optimization.penalty import get_penalties
from optimus_core import load_obj
from recommend import (
    bulk_optimize,
    extract_optimization_results_from_solvers,
    generate_run_id,
    make_solver,
    make_stopper,
)


def get_recommendations() -> Pipeline:
    return Pipeline(
        [
            node(
                make_objective,
                inputs=dict(
                    models_dict="models_dict", parameters="params:opt_upstream"
                ),
                outputs="objective",
            ),
            node(
                load_obj,
                inputs="params:solver_class",
                outputs="solver_class",
            ),
            node(
                load_obj,
                inputs="params:stopper_class",
                outputs="stopper_class",
            ),
            node(
                get_penalties,
                inputs=dict(models_dict="models_dict"),
                outputs="penalty_func",
            ),
            node(
                make_problem,
                inputs=dict(
                    data="data",
                    td="td",
                    penalties="penalty_func",
                    objective="objective",
                    domain_generator="params:domain_generator",
                    sense="params:sense",
                    areas_to_optimize="params:areas_to_optimize",
                ),
                outputs=["problem_dict", "bounds_dict_pre", "control_off_indexes"],
                name="make_problem",
            ),
            node(
                adjust_bounds_dict,
                inputs=dict(
                    data="data",
                    td="td",
                    bounds_dict="bounds_dict_pre",
                    areas_to_optimize="params:areas_to_optimize",
                    bounds_update="params:bounds_update",
                ),
                outputs="bounds_dict",
            ),
            node(
                make_solver,
                inputs={
                    "solver_class": "solver_class",
                    "solver_kwargs": "params:solver",
                    "domain_dict": "bounds_dict",
                },
                outputs="solver_dict",
                name="make_solver",
            ),
            node(
                make_stopper,
                inputs={
                    "stopper_class": "stopper_class",
                    "stopper_kwargs": "params:stopper",
                },
                outputs="stopper",
                name="make_stopper",
            ),
            node(
                bulk_optimize,
                inputs={
                    "problem_dict": "problem_dict",
                    "solver_dict": "solver_dict",
                    "stopper": "stopper",
                    "n_jobs": "params:n_jobs",
                },
                outputs="solver_dict_after_optimization",
                name="bulk_optimizer",
            ),
            node(
                extract_optimization_results_from_solvers,
                inputs={
                    "problem_dict": "problem_dict",
                    "solver_dict": "solver_dict_after_optimization",
                },
                outputs="optim_results_solver",
            ),
            node(
                add_model_predictions,
                inputs=dict(
                    models_dict="models_dict", optim_results="optim_results_solver"
                ),
                outputs="optim_results",
            ),
            node(
                generate_run_id,
                inputs="optim_results",
                outputs="recommend_results",
                name="generate_run_id",
            ),
        ],
    ).tag("recommend")
