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
This is a SCRIPT for updating some datasets' artefacts.
Use it when structure of the module changes and pickles stop working.
"""

import logging
import pickle
import typing as tp

import pandas as pd

from optimizer import penalty, repair
from optimizer.solvers import DifferentialEvolutionSolver
from optimizer.stoppers import NoImprovementStopper
from optimus_core import partial_wrapper
from recommend import (
    MinMaxDomainGenerator,
    bulk_optimize,
    extract_optimization_results_from_solvers,
    make_problem,
    make_solver,
)
from recommend.datasets import (
    DATA_DIR,
    get_sample_optimization_explainer_tag_dict,
    get_sample_recommend_input_data,
    get_trained_model,
)
from recommend.datasets.sample_class_definitions import (
    FlowPenalty,
    OrePulpPhRepair,
    SilicaObjective,
    repair_column_by_setting_value,
)

logger = logging.getLogger(__name__)

FIRST_REPAIR_CONSTRAINT_VALUE = 10.0
SECOND_REPAIR_CONSTRAINT_VALUE = 9.6
FIRST_PENALTY_CONSTRAINT_VALUE = 4000
FIRST_PENALTY_MULTIPLIER = 0.0125


def main() -> None:
    flow_penalty = penalty(  # noqa: WPS317
        FlowPenalty(),
        ">=",
        FIRST_PENALTY_CONSTRAINT_VALUE,
        name="starch_and_amina_flow",
        penalty_multiplier=FIRST_PENALTY_MULTIPLIER,
    )
    first_ore_pulp_ph_repair = repair(
        OrePulpPhRepair(),
        "<=",
        FIRST_REPAIR_CONSTRAINT_VALUE,
        repair_function=partial_wrapper(
            repair_column_by_setting_value,
            column="ore_pulp_ph",
            value_to_set=FIRST_REPAIR_CONSTRAINT_VALUE,
        ),
        name=f"ore_pulp_ph =< {FIRST_REPAIR_CONSTRAINT_VALUE:0.1f}",
    )
    second_ore_pulp_ph_repair = repair(  # noqa: WPS317
        OrePulpPhRepair(),
        ">=",
        SECOND_REPAIR_CONSTRAINT_VALUE,
        repair_function=partial_wrapper(
            repair_column_by_setting_value,
            column="ore_pulp_ph",
            value_to_set=SECOND_REPAIR_CONSTRAINT_VALUE,
        ),
        name=f"ore_pulp_ph >= {SECOND_REPAIR_CONSTRAINT_VALUE:0.1f}",
    )
    problem_dict, domain_dict = make_problem(
        data=get_sample_recommend_input_data(),
        td=get_sample_optimization_explainer_tag_dict(),
        objective=SilicaObjective(get_trained_model()),
        penalties=flow_penalty,
        repairs=[first_ore_pulp_ph_repair, second_ore_pulp_ph_repair],
        sense="minimize",
        domain_generator=MinMaxDomainGenerator,
    )
    solver_dict = make_solver(
        solver_class=DifferentialEvolutionSolver,
        solver_kwargs={
            "sense": "minimize",
            "seed": 0,
            "maxiter": 100,
            "mutation": [0.5, 1.0],
            "recombination": 0.7,
            "strategy": "best1bin",
        },
        domain_dict=domain_dict,
    )
    solvers_after_optimization = bulk_optimize(
        problem_dict=problem_dict,
        solver_dict=solver_dict,
        stopper=NoImprovementStopper(
            patience=100,
            sense="minimize",
            min_delta=0.1,
        ),
        n_jobs=8,
    )
    optimization_results = extract_optimization_results_from_solvers(
        solvers_after_optimization,
        problem_dict,
    )
    _set_pickle_data(problem_dict, "sample_problem_dict")
    _set_pickle_data(solvers_after_optimization, "sample_solver_dict")
    _set_pickle_data(optimization_results, "sample_recommend_results")


def _dump_dataset(
    dataset: pd.DataFrame,
    file_name: str,
    **kwargs: tp.Any,
) -> None:
    return dataset.to_csv(DATA_DIR / f"{file_name}.csv", **kwargs)


def _set_pickle_data(data: tp.Any, file_name: str) -> None:
    with open(DATA_DIR / f"{file_name}.pkl", "wb") as fw:
        pickle.dump(data, fw)


if __name__ == "__main__":
    main()
