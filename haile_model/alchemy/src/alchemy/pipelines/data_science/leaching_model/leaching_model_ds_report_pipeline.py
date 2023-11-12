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
    get_2d_ore_cluster_plot,
    get_3d_ore_cluster_plot,
    get_box_plots,
    get_ore_cluster_profile_for_train_data,
    model_report_information,
)


def create_model_report_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                model_report_information,
                inputs={"params": "params:report"},
                outputs=None,
                name="report_information",
            ),
            node(
                get_ore_cluster_profile_for_train_data,
                inputs={
                    "train_data": "train_data_without_nulls",
                    "params": "params:report",
                    "td": "td",
                },
                outputs="ore_cluster_profile_train_data",
                name="ore_cluster_profile_for_train_data",
            ),
            node(
                get_2d_ore_cluster_plot,
                inputs={
                    "params": "params:report",
                    "td": "td",
                    "data_scaled": "train_data_upstream_cluster_scaled",
                    "first_level_cluster_trained_model": (
                        "first_level_cluster_trained_model"
                    ),
                },
                outputs="ore_cluster_2d",
                name="get_2d_ore_cluster_plot",
            ),
            node(
                get_3d_ore_cluster_plot,
                inputs={
                    "params": "params:report",
                    "td": "td",
                    "data_scaled": "train_data_upstream_cluster_scaled",
                    "first_level_cluster_trained_model": (
                        "first_level_cluster_trained_model"
                    ),
                },
                outputs="ore_cluster_3d",
                name="get_3d_ore_cluster_plot",
            ),
            node(
                get_box_plots,
                inputs={
                    "train_data": "train_data_without_nulls",
                    "params": "params:report",
                    "td": "td",
                },
                outputs="get_ore_cluster_box_plots",
                name="get_ore_cluster_box_plots",
            ),
        ],
    ).tag(["modeling", "reporting"])
