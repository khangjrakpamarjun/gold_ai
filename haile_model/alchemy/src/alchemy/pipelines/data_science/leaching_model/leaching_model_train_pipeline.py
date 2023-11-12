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

from alchemy.pipelines.data_science.leaching_model.leaching_model_train_nodes import (
    get_first_level_clusters,
    get_operating_mode_per_cluster,
)


def create_train_model_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=get_first_level_clusters,
                inputs=dict(
                    data="data",
                    td="td",
                    params=f"params:train_model",
                ),
                outputs=[
                    "train_data_upstream_cluster",
                    "train_data_upstream_cluster_scaled",
                    "train_data_all_tags",
                    "first_cluster_trained_model",
                    "model_features_first_level_cluster",
                ],
                name="get_first_level_clusters",
            ),
            node(
                func=get_operating_mode_per_cluster,
                inputs=dict(
                    data_upstream_cluster="train_data_upstream_cluster",
                    td="td",
                    df_shift="train_data_all_tags",
                    params=f"params:train_model",
                ),
                outputs=[
                    "train_data_with_clusters",
                    "best_operating_mode_per_cluster",
                    "second_cluster_trained_models",
                    "recovery_per_cluster",
                    "model_features",
                    "all_operating_mode_per_cluster",
                    "model_features_second_level_cluster",
                    "model_features_second_level_cluster_dict",
                    "data_process_scaled",
                ],
                name="get_operating_mode_per_cluster",
            ),
        ],
    ).tag("model_training")
