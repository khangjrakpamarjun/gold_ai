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

from alchemy.pipelines.optimization.generate_recommendations.export_recommendations_nodes import (
    adjust_for_fe_full_circuit,
    adjust_for_fe_upstream,
)
from alchemy.pipelines.optimization.optimization.optimization_nodes import (
    prepare_runs,
    prepare_upstream_recommendations,
)
from optimus_core.utils import partial_wrapper
from recommend.export import prepare_predictions, prepare_states, prepare_tags


def prepare_recommendations() -> Pipeline:
    return Pipeline(
        nodes=[
            node(
                prepare_predictions,
                inputs={"scores": "recommend_results_translated"},
                outputs="cra_predictions",
            ),
            node(
                partial_wrapper(
                    prepare_upstream_recommendations,
                    default_status="Pending",
                    on_control_only=True,
                ),
                inputs={
                    "scores": "recommend_results_translated",
                    "td": "td",
                    "solver_dict": "solver_dict",
                    "areas_to_optimize": "params:areas_to_optimize",
                    "translation_layer_tags": "params:translation_layer_output_tags",
                    "control_tags_not_for_ui": "params:control_tags_not_for_ui",
                },
                outputs="cra_recommendations",
            ),
            node(
                partial_wrapper(
                    prepare_runs,
                    iso_format="%Y-%m-%dT%H:%M:%SZ",  # noqa: WPS323
                    timestamp_col="timestamp",
                ),
                inputs="recommend_results_translated",
                outputs="cra_runs",
            ),
            node(prepare_tags, inputs="td", outputs="cra_tags"),
            node(
                prepare_states,
                inputs={
                    "scores": "recommend_results_translated",
                    "ui_states": "params:ui_states",
                },
                outputs="cra_states",
            ),
        ],
    ).tag("prepare_recs")


def export_full_circuit_recommendations(
    upstream_model_name: str,
    downstream_model_name: str,
) -> Pipeline:
    return Pipeline(
        nodes=[
            node(
                func=adjust_for_fe_full_circuit,
                inputs=dict(
                    recommend_results_translated=(
                        f"{upstream_model_name}.recommend_results_translated"
                    ),
                    upstream_runs=f"{upstream_model_name}.cra_runs",
                    upstream_recommendations=(
                        f"{upstream_model_name}.cra_recommendations"
                    ),
                    upstream_predictions=f"{upstream_model_name}.cra_predictions",
                    downstream_recommendations=(
                        f"{downstream_model_name}.recommendations_live"
                    ),
                    downstream_predictions=f"{downstream_model_name}.predictions",
                    kpis_for_ui="params:ui.kpi_for_ui",
                    kpis_for_db="params:ui.kpi_last_calc_value_full_circuit",
                    td="td",
                    pi_data="pi_data_ingested",
                    pi_with_derived_tags="df_merged_with_derived_features",
                    baseline_tph=f"{upstream_model_name}.baseline_tph",
                ),
                outputs=dict(
                    runs="runs",
                    recommendations="recommendations",
                    predictions="predictions",
                    kpi_last_calculated_value="last_calculated_value",
                ),
                namespace="create_live_recommendations",
                name="adjust_for_fe_full_circuit",  # TODO: node names not exposed, need to be able to debug single node
            ),
        ],
    ).tag("export")


def export_upstream_recommendations(
    upstream_model_name: str,
) -> Pipeline:
    return Pipeline(
        nodes=[
            node(
                func=adjust_for_fe_upstream,
                inputs=dict(
                    recommend_results_translated=(
                        f"{upstream_model_name}.recommend_results_translated"
                    ),
                    upstream_runs=f"{upstream_model_name}.cra_runs",
                    upstream_recommendations=(
                        f"{upstream_model_name}.cra_recommendations"
                    ),
                    upstream_predictions=f"{upstream_model_name}.cra_predictions",
                    kpis_for_ui="params:ui.kpi_for_ui",
                    kpis_for_db="params:ui.kpi_last_calc_value_upstream_circuit",
                    td="td",
                    pi_data="pi_data_ingested",
                    pi_with_derived_tags="df_merged_with_derived_features",
                    baseline_tph=f"{upstream_model_name}.baseline_tph",
                ),
                outputs=dict(
                    runs="runs",
                    recommendations="recommendations",
                    predictions="predictions",
                    kpi_last_calculated_value="last_calculated_value",
                ),
                namespace="create_live_recommendations",
            ),
        ],
    ).tag("export")
