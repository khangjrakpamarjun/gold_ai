from kedro.pipeline import Pipeline, pipeline

from alchemy.pipelines.data_science.leaching_model import (
    leaching_model_ds_report_pipeline,
    leaching_model_train_pipeline,
)
from alchemy.pipelines.data_science.model_input import model_input_pipeline
from alchemy.pipelines.data_science.model_report import model_report_pipeline
from alchemy.pipelines.data_science.model_train import model_train_pipeline


def create_pipeline() -> Pipeline:
    model_sag = create_base_model_pipeline(model_name="upstream.modeling.sag_power")
    model_flotation = create_base_model_pipeline(
        model_name="upstream.modeling.sulphide_grade"
    )
    model_mass_pull = create_base_model_pipeline(
        model_name="upstream.modeling.mass_pull"
    )
    model_cil_recovery = create_leaching_model_pipeline(
        model_name="downstream.modeling.cil_recovery"
    )
    return model_sag + model_flotation + model_mass_pull + model_cil_recovery


def create_base_model_pipeline(model_name: str) -> Pipeline:
    model_input = model_input_pipeline.create_model_input_pipeline()
    train_model = model_train_pipeline.create_train_model_pipeline()
    model_report = model_report_pipeline.create_model_report_pipeline()

    model = (
        pipeline(
            pipe=model_input,
            inputs={"data": "df_merged_with_features_outlier_clean", "td": "td"},
            parameters={"params:model_input": f"params:{model_name}.model_input"},
            namespace=f"{model_name}",
        )
        + pipeline(
            pipe=train_model,
            inputs={"td": "td"},
            parameters={"params:train_model": f"params:{model_name}.train_model"},
            namespace=f"{model_name}",
        )
        + pipeline(
            pipe=model_report,
            inputs={"td": "td"},
            parameters={"params:report": f"params:{model_name}.report"},
            namespace=f"{model_name}",
        )
    )

    model = model.tag(model_name)
    return model


def create_leaching_model_pipeline(model_name: str) -> Pipeline:
    model_input = model_input_pipeline.create_model_input_pipeline()
    train_model = leaching_model_train_pipeline.create_train_model_pipeline()
    model_report = leaching_model_ds_report_pipeline.create_model_report_pipeline()

    model = (
        pipeline(
            pipe=model_input,
            inputs={"data": "df_merged_with_features_outlier_clean", "td": "td"},
            parameters={"params:model_input": f"params:{model_name}.model_input"},
            namespace=f"{model_name}",
        )
        + pipeline(
            pipe=train_model,
            inputs={"data": f"{model_name}.train_data", "td": "td"},
            parameters={
                "params:train_model": f"params:{model_name}.train_model",
            },
            namespace=f"{model_name}",
        )
        + pipeline(
            pipe=model_report,
            inputs={
                "train_data_without_nulls": f"{model_name}.train_data_with_clusters",
                "first_level_cluster_trained_model": (
                    f"{model_name}.first_cluster_trained_model"
                ),
                "train_data_upstream_cluster_scaled": (
                    f"{model_name}.train_data_upstream_cluster_scaled"
                ),
                "td": "td",
            },
            parameters={"params:report": f"params:{model_name}.report"},
            namespace=f"{model_name}",
        )
    )

    model = model.tag(model_name)
    return model


# TODO: Items:
# Train test predictions to have timestamp in csvs
# Get names of features on shap importance and feature importance csv
# Catalog entry to update atuomatically based on model_name passed
# Pass whole params of model in one go with model_name
