import logging
from typing import Dict

from kedro.pipeline import Pipeline, node, pipeline

from alchemy.pipelines.optimization.generate_recommendations.export_recommendations_pipeline import (
    export_full_circuit_recommendations,
    export_upstream_recommendations,
    prepare_recommendations,
)
from alchemy.pipelines.optimization.generate_recommendations.generate_recommendations_pipeline import (
    get_recommendations,
)
from alchemy.pipelines.optimization.leaching_model import (
    leaching_model_opt_report_pipeline,
    leaching_recommendations_pipeline,
)
from alchemy.pipelines.optimization.optimization.optimization_model_input import (
    create_model_input_pipeline,
)
from alchemy.pipelines.optimization.optimization.optimization_nodes import (
    combine_cfa_results,
    create_models_dict,
    get_baseline_tph,
    inject_upstream_opt,
    translation_layer,
)

logger = logging.getLogger(__name__)


def create_recommendations_pipeline(
    env: str,
    upstream_pipeline: Pipeline,
    downstream_pipeline: Pipeline,
    export_recs_pipeline: Pipeline,
):
    # If environment is base - no exporting required in db tables (Counterfactual scenario)
    circuit_pipeline = upstream_pipeline + downstream_pipeline + export_recs_pipeline

    # If live environment then export recommendations to db tables for UI
    if env == "live":
        circuit_pipeline = (
            upstream_pipeline + downstream_pipeline + export_recs_pipeline
        )

    # If none of the two environments then throw error
    elif env != "base":
        msg = "Unexpected value for ´env´ arg. Accepted values are ´base´ and ´live´"
        logger.error(msg)
        raise ValueError(msg)

    return circuit_pipeline


def create_upstream_export_pipeline(
    env: str,
    upstream_pipeline: Pipeline,
    export_recs_pipeline: Pipeline = None,
):
    # If environment is base - no exporting required in db tables (Counterfactual scenario)
    circuit_pipeline = upstream_pipeline

    # If live environment then export recommendations to db tables for UI
    if env == "live":
        circuit_pipeline = upstream_pipeline + export_recs_pipeline

    # If none of the two environments then throw error
    elif env != "base":
        msg = "Unexpected value for ´env´ arg. Accepted values are ´base´ and ´live´"
        logger.error(msg)
        raise ValueError(msg)

    return circuit_pipeline


def create_upstream_pipeline(testing: bool = False):
    """
    Optimization pipeline for upstream process - Grinding and Flotation
    Create recommendations, run ids and overall KPI stats

    Returns: Recommendations for upstream circuit

    """
    models_dict = dict(
        sag_power="upstream.modeling.sag_power.trained_model",
        sulphide_grade="upstream.modeling.sulphide_grade.trained_model",
        mass_pull="upstream.modeling.mass_pull.trained_model",
    )

    opt_upstream = create_upstream_opt_pipeline(
        model_name="upstream.optimisation.opt_upstream",
        models_dict=models_dict,
        baseline_model_name="upstream.baselining.baseline_tph",
        is_testing=testing,
    )

    return opt_upstream


def create_upstream_opt_pipeline(
    model_name: str,
    models_dict: Dict,
    baseline_model_name: str,
    is_testing: bool = False,
):
    model_input = create_model_input_pipeline()
    opt_pipe = Pipeline(
        [
            node(
                func=create_models_dict,
                inputs=models_dict,
                outputs=f"{model_name}.models_dict",
                name="create_models_dict",
            ),
            *(
                [
                    pipeline(
                        pipe=get_recommendations(),
                        inputs={"data": "test_data", "td": "td"},
                        parameters={f"params:opt_upstream": f"params:{model_name}"},
                        namespace=model_name,
                    ),
                ]
                if is_testing
                else [
                    pipeline(
                        pipe=model_input,
                        inputs={
                            "data": "df_merged_with_features_outlier_clean",
                            "td": "td",
                        },
                        parameters={
                            "params:model_input": f"params:{model_name}.model_input"
                        },
                        namespace=model_name,
                    ),
                    pipeline(
                        pipe=get_recommendations(),
                        inputs={
                            "data": f"{model_name}.data_filtered_by_target",
                            "td": "td",
                        },
                        parameters={f"params:opt_upstream": f"params:{model_name}"},
                        namespace=model_name,
                    ),
                ]
            ),
            node(
                get_baseline_tph,
                inputs=dict(
                    baseline_tph_model=f"{baseline_model_name}.baseline_tph_model",
                    baseline_per_cluster=f"{baseline_model_name}.baseline_per_cluster",
                    optim_results=f"{model_name}.optim_results",
                    baseline_features_median=(
                        f"{baseline_model_name}.baseline_features_median"
                    ),
                ),
                outputs=f"{model_name}.baseline_tph",
            ),
            node(
                translation_layer,
                inputs=dict(
                    scores=f"{model_name}.recommend_results",
                    params=f"params:{model_name}.translation_layer",
                ),
                outputs=f"{model_name}.recommend_results_translated",
            ),
            pipeline(
                pipe=prepare_recommendations(),
                inputs={"td": "td"},
                namespace=model_name,
            ),
        ]
    )
    opt_pipe = opt_pipe.tag([model_name])
    return opt_pipe


def create_downstream_pipeline(testing: bool = False):
    """
    Downstream pipeline to prepare recommendations from leaching circuit

    Returns: Downstream pipeline recommendations

    """
    opt_downstream = create_downstream_opt_pipeline(
        downstream_model_name="downstream.modeling.cil_recovery",
        downstream_opt_model_name="downstream.optimisation.opt_downstream",
        upstream_opt_model_name="upstream.optimisation.opt_upstream",
        baseline_model_name="downstream.baselining.baseline_cil_recovery",
        is_testing=testing,
    )

    return opt_downstream


def create_downstream_opt_pipeline(
    downstream_model_name: str,
    downstream_opt_model_name: str,
    upstream_opt_model_name: str,
    baseline_model_name: str,
    is_testing: bool = False,
) -> Pipeline:
    model_input = create_model_input_pipeline()
    recommendations = leaching_recommendations_pipeline.get_recommendations()
    recommendations_report = (
        leaching_model_opt_report_pipeline.create_opt_report_pipeline()
    )

    opt_pipe = Pipeline(
        [
            *(
                [
                    node(
                        func=inject_upstream_opt,
                        inputs=[
                            f"{upstream_opt_model_name}.optim_results",
                            "test_data",
                        ],
                        outputs="data_upstream_optimized",
                    ),
                    pipeline(
                        pipe=recommendations,
                        inputs={
                            "data": "test_data",
                            "all_operating_mode_per_cluster": (
                                f"{downstream_model_name}.all_operating_mode_per_cluster"
                            ),
                            "train_df_processed_with_clusters": (
                                f"{downstream_model_name}.train_data_with_clusters"
                            ),
                            "test_data": (
                                "data_upstream_optimized"
                            ),  # TODO: clarify why this naming
                            "data_upstream_optimized": (
                                "data_upstream_optimized"
                            ),  # TODO: remove above duplication
                            "median_values_for_tags": (
                                f"{baseline_model_name}.median_values_for_tags"
                            ),
                            "td": "td",
                            "best_operating_mode_per_cluster": (
                                f"{downstream_model_name}.best_operating_mode_per_cluster"
                            ),
                            "first_cluster_trained_model": (
                                f"{downstream_model_name}.first_cluster_trained_model"
                            ),
                            "second_cluster_trained_models": (
                                f"{downstream_model_name}.second_cluster_trained_models"
                            ),
                            "model_features_first_level_cluster": (
                                f"{downstream_model_name}.model_features_first_level_cluster"
                            ),
                            "model_features_second_level_cluster": (
                                f"{downstream_model_name}.model_features_second_level_cluster"
                            ),
                            "model_features_second_level_cluster_dict": f"{downstream_model_name}.model_features_second_level_cluster_dict",
                            "flc_trained_model_hist_baseline": (
                                f"{baseline_model_name}.flc_trained_model"
                            ),
                            "slc_trained_models_hist_baseline": (
                                f"{baseline_model_name}.slc_trained_models"
                            ),
                            "model_features_flc_hist_baseline": (
                                f"{baseline_model_name}.model_features_flc"
                            ),
                            "baseline_historic_upstream_and_downstream": (
                                "baseline_historic.baseline_historic_upstream_and_downstream"
                            ),
                            "model_features_slc_dict_hist_baseline": (
                                f"{baseline_model_name}.model_features_slc_dict"
                            ),
                            "baseline_tph": f"{upstream_opt_model_name}.baseline_tph",
                        },
                        parameters={
                            "params:train_model": (
                                f"params:{downstream_opt_model_name}.train_model"
                            ),
                        },
                        namespace=downstream_opt_model_name,
                    ),
                ]
                if is_testing
                else [
                    pipeline(
                        pipe=model_input,
                        inputs={
                            "data": "df_merged_with_features_outlier_clean",
                            "td": "td",
                        },
                        parameters={
                            "params:model_input": (
                                f"params:{downstream_opt_model_name}.model_input"
                            )
                        },
                        namespace=downstream_opt_model_name,
                    ),
                    node(
                        func=inject_upstream_opt,
                        inputs=[
                            f"{upstream_opt_model_name}.optim_results",
                            f"{downstream_opt_model_name}.data_filtered_by_target",
                        ],
                        outputs="data_upstream_optimized",
                    ),
                    pipeline(
                        pipe=recommendations,
                        inputs={
                            "data": f"{downstream_opt_model_name}.data_filtered_by_target",
                            "all_operating_mode_per_cluster": (
                                f"{downstream_model_name}.all_operating_mode_per_cluster"
                            ),
                            "train_df_processed_with_clusters": (
                                f"{downstream_model_name}.train_data_with_clusters"
                            ),
                            "test_data": (
                                "data_upstream_optimized"
                            ),  # TODO: clarify why this naming
                            "data_upstream_optimized": (
                                "data_upstream_optimized"
                            ),  # TODO: remove above duplication
                            "median_values_for_tags": (
                                f"{baseline_model_name}.median_values_for_tags"
                            ),
                            "td": "td",
                            "best_operating_mode_per_cluster": (
                                f"{downstream_model_name}.best_operating_mode_per_cluster"
                            ),
                            "first_cluster_trained_model": (
                                f"{downstream_model_name}.first_cluster_trained_model"
                            ),
                            "second_cluster_trained_models": (
                                f"{downstream_model_name}.second_cluster_trained_models"
                            ),
                            "model_features_first_level_cluster": (
                                f"{downstream_model_name}.model_features_first_level_cluster"
                            ),
                            "model_features_second_level_cluster": (
                                f"{downstream_model_name}.model_features_second_level_cluster"
                            ),
                            "model_features_second_level_cluster_dict": f"{downstream_model_name}.model_features_second_level_cluster_dict",
                            "flc_trained_model_hist_baseline": (
                                f"{baseline_model_name}.flc_trained_model"
                            ),
                            "slc_trained_models_hist_baseline": (
                                f"{baseline_model_name}.slc_trained_models"
                            ),
                            "model_features_flc_hist_baseline": (
                                f"{baseline_model_name}.model_features_flc"
                            ),
                            "baseline_historic_upstream_and_downstream": (
                                "baseline_historic.baseline_historic_upstream_and_downstream"
                            ),
                            "model_features_slc_dict_hist_baseline": (
                                f"{baseline_model_name}.model_features_slc_dict"
                            ),
                            "baseline_tph": f"{upstream_opt_model_name}.baseline_tph",
                        },
                        parameters={
                            "params:train_model": (
                                f"params:{downstream_opt_model_name}.train_model"
                            ),
                        },
                        namespace=downstream_opt_model_name,
                    ),
                ]
            ),
            pipeline(
                pipe=recommendations_report,
                inputs={
                    "recommendations_cfa": (
                        f"{downstream_opt_model_name}.recommendations_cfa"
                    ),
                    "td": "td",
                },
                parameters={
                    "params:report": f"params:{downstream_opt_model_name}.report"
                },
                namespace=downstream_opt_model_name,
            ),
        ]
    )
    opt_pipe = opt_pipe.tag([downstream_opt_model_name])

    return opt_pipe


def create_export_recs_pipeline(full_circuit: bool = True):
    if full_circuit:
        ex_pipe = Pipeline(
            [
                pipeline(
                    pipe=export_full_circuit_recommendations(
                        upstream_model_name="upstream.optimisation.opt_upstream",
                        downstream_model_name="downstream.optimisation.opt_downstream",
                    ),
                )
            ]
        )
    else:
        ex_pipe = Pipeline(
            [
                pipeline(
                    pipe=export_upstream_recommendations(
                        upstream_model_name="upstream.optimisation.opt_upstream",
                    ),
                )
            ]
        )

    return ex_pipe


def create_export_cf_pipeline():
    """
    Downstream pipeline to prepare recommendations from leaching circuit

    Returns: Downstream pipeline recommendations

    """
    get_export_cf_pipeline = export_cf_pipeline(
        downstream_opt_model_name="downstream.optimisation.opt_downstream",
        upstream_opt_model_name="upstream.optimisation.opt_upstream",
    )

    return get_export_cf_pipeline


def export_cf_pipeline(downstream_opt_model_name: str, upstream_opt_model_name: str):
    export_cf_pipe = Pipeline(
        [
            node(
                func=combine_cfa_results,
                inputs={
                    "cfa_upstream": (
                        f"{upstream_opt_model_name}.recommend_results_translated"
                    ),
                    "cfa_downstream": (
                        f"{downstream_opt_model_name}.recommendations_cfa"
                    ),
                    "td": "td",
                    "areas_to_optimize": (
                        f"params:{upstream_opt_model_name}.areas_to_optimize"
                    ),
                    "target_tags": f"params:{upstream_opt_model_name}.target_tags",
                },
                outputs="combined_cfa_results",
            )
        ]
    )
    return export_cf_pipe
