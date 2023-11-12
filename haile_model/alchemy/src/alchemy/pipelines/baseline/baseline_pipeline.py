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

from kedro.pipeline import Pipeline, node, pipeline

from alchemy.pipelines.baseline.baseline_nodes import (
    baseline_historic_upstream_and_downstream,
    baseline_tph_model,
    get_median_for_baseline_recovery,
)
from alchemy.pipelines.data_science.leaching_model import (
    leaching_model_ds_report_pipeline,
)
from alchemy.pipelines.data_science.leaching_model.leaching_model_train_nodes import (
    get_first_level_clusters,
    get_operating_mode_per_cluster,
)
from alchemy.pipelines.data_science.model_input.model_input_nodes import (
    aggregate_data,
    filter_data_by_target,
    filter_data_by_timestamp,
)


def create_model_input_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=aggregate_data,
                inputs=[
                    "data",
                    "td",
                    "params:model_input.aggregation",
                ],
                outputs="df_aggregated",
                name="aggregate_data",
            ),
            node(
                func=filter_data_by_timestamp,
                inputs=dict(data="df_aggregated", params="params:model_input"),
                outputs="data_filtered_by_timestamp",
                name="filter_by_timestamp",
            ),
            node(
                func=filter_data_by_target,
                inputs=dict(
                    params=f"params:model_input",
                    td="td",
                    data="data_filtered_by_timestamp",
                ),
                outputs="data_filtered_by_target",
                name="filter_target",
            ),
        ],
    ).tag("model_input")


def create_leaching_baseline_pipeline(model_name: str) -> Pipeline:
    model_input = create_model_input_pipeline()
    model_report = leaching_model_ds_report_pipeline.create_model_report_pipeline()
    leach_base_pipe = Pipeline(
        [
            pipeline(
                pipe=model_input,
                inputs={"data": "df_merged_with_features_outlier_clean", "td": "td"},
                parameters={"params:model_input": f"params:{model_name}.model_input"},
                namespace=model_name,
            ),
            node(
                func=get_median_for_baseline_recovery,
                inputs=dict(
                    data=f"{model_name}.data_filtered_by_target",
                    td="td",
                    params=f"params:{model_name}.train_model",
                ),
                outputs=f"{model_name}.median_values_for_tags",
                name="downstream_model_median_values_for_tags",
                namespace=model_name,
            ),
            node(
                func=get_first_level_clusters,
                inputs=dict(
                    data=f"{model_name}.data_filtered_by_target",
                    td="td",
                    params=f"params:{model_name}.train_model",
                ),
                outputs=[
                    f"{model_name}.train_data_upstream_cluster",
                    f"{model_name}.train_data_upstream_cluster_scaled",
                    f"{model_name}.train_data_all_tags",
                    f"{model_name}.flc_trained_model",
                    f"{model_name}.model_features_flc",
                ],
                name="baseline_recovery_first_level_clusters",
                namespace=model_name,
            ),
            node(
                func=get_operating_mode_per_cluster,
                inputs=dict(
                    data_upstream_cluster=f"{model_name}.train_data_upstream_cluster",
                    td="td",
                    df_shift=f"{model_name}.train_data_all_tags",
                    params=f"params:{model_name}.train_model",
                ),
                outputs=[
                    f"{model_name}.train_data_with_clusters",
                    f"{model_name}.best_operating_mode_per_cluster",
                    f"{model_name}.slc_trained_models",
                    "recovery_per_cluster",
                    "model_features",
                    f"{model_name}.baseline_per_cluster_per_operating_mode",
                    f"{model_name}.model_features_slc",
                    f"{model_name}.model_features_slc_dict",
                    "data_process_scaled",
                ],
                name="baseline_recovery_second_level_clusters",
                namespace=model_name,
            ),
            pipeline(
                pipe=model_report,
                inputs={
                    "train_data_without_nulls": (
                        f"{model_name}.train_data_with_clusters"
                    ),
                    "first_level_cluster_trained_model": (
                        f"{model_name}.flc_trained_model"
                    ),
                    "train_data_upstream_cluster_scaled": (
                        f"{model_name}.train_data_upstream_cluster_scaled"
                    ),
                    "td": "td",
                },
                parameters={"params:report": f"params:{model_name}.report"},
                namespace=model_name,
            ),
        ]
    )
    leach_base_pipe = leach_base_pipe.tag([model_name])
    return leach_base_pipe


def create_tph_baseline_pipeline(model_name: str) -> Pipeline:
    model_input = create_model_input_pipeline()
    tph_base_pipe = Pipeline(
        [
            pipeline(
                pipe=model_input,
                inputs={"data": "df_merged_with_features_outlier_clean", "td": "td"},
                parameters={"params:model_input": f"params:{model_name}.model_input"},
                namespace=model_name,
            ),
            node(
                func=baseline_tph_model,
                inputs=dict(
                    data=f"{model_name}.data_filtered_by_target",
                    td="td",
                    params=f"params:{model_name}",
                ),
                outputs=[
                    f"{model_name}.baseline_per_cluster",
                    f"{model_name}.baseline_tph_model",
                    f"{model_name}.baseline_features_median",
                ],
                name="baseline_tph_model",
                namespace=model_name,
            ),
        ]
    )
    tph_base_pipe = tph_base_pipe.tag([model_name])

    return tph_base_pipe


def create_combined_baseline_pipeline(
    upstream_baseline_model_name: str,
    downstream_baseline_model_name: str,
    model_name: str,
) -> Pipeline:
    combined_base_pipe = Pipeline(
        [
            node(
                func=baseline_historic_upstream_and_downstream,
                inputs=dict(
                    baseline_tph=f"{upstream_baseline_model_name}.baseline_per_cluster",
                    baseline_cil_recovery=f"{downstream_baseline_model_name}.baseline_per_cluster_per_operating_mode",
                    td="td",
                ),
                outputs=f"{model_name}.baseline_historic_upstream_and_downstream",
                name="get_hist_baselines_upstream_and_downstream",
                namespace=model_name,
            ),
        ]
    )
    combined_base_pipe = combined_base_pipe.tag([model_name])
    return combined_base_pipe


def create_baselines_pipeline() -> Pipeline:
    upstream_baseline_model_name = "upstream.baselining.baseline_tph"
    downstream_baseline_model_name = "downstream.baselining.baseline_cil_recovery"

    baseline_tph = create_tph_baseline_pipeline(model_name=upstream_baseline_model_name)
    baseline_cil_recovery = create_leaching_baseline_pipeline(
        model_name=downstream_baseline_model_name
    )

    combined_pipe = create_combined_baseline_pipeline(
        upstream_baseline_model_name=upstream_baseline_model_name,
        downstream_baseline_model_name=downstream_baseline_model_name,
        model_name="baseline_historic",
    )
    baseline_historic_pipe = baseline_tph + baseline_cil_recovery + combined_pipe

    return baseline_historic_pipe
