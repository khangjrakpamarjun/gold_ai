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

from alchemy.pipelines.data_science.leaching_model.leaching_model_report_nodes import (
    get_box_plots_cfa,
    get_ore_cluster_profile_for_cfa,
    model_report_information,
)


def create_opt_report_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                model_report_information,
                inputs={"params": "params:report"},
                outputs=None,
                name="report_information",
            ),
            node(
                get_ore_cluster_profile_for_cfa,
                inputs={
                    "train_data": "recommendations_cfa",
                    "params": "params:report",
                    "td": "td",
                },
                outputs="ore_cluster_profile_cfa",
                name="ore_cluster_profile_for_cfa",
            ),
            node(
                get_box_plots_cfa,
                inputs={
                    "train_data": "recommendations_cfa",
                    "params": "params:report",
                    "td": "td",
                },
                outputs="get_ore_cluster_box_plots_cfa",
                name="get_ore_cluster_box_plots_cfa",
            ),
        ],
    ).tag(["optimization", "reporting"])
