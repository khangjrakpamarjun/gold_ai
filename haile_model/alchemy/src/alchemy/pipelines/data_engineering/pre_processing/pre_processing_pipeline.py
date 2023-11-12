from kedro.pipeline import Pipeline, node, pipeline

from alchemy.pipelines.data_engineering.pre_processing.pre_processing_nodes import (
    clean_data,
    merge_datasets,
    remove_outliers,
)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=merge_datasets,
                inputs=dict(pi_data_ingested="pi_data_ingested"),
                outputs="df_merged",
                name="merge_datasets",
            ),
            node(
                func=clean_data,
                inputs=[
                    "df_merged",
                    "td",
                    "params:remove_outliers_merged_data",
                ],
                outputs="df_merged_clean",
                name="clean_data",
            ),
            node(
                func=remove_outliers,
                inputs=[
                    "df_merged_clean",
                    "td",
                    "params:remove_outliers_merged_data",
                ],
                outputs=dict(
                    df="df_merged_outlier_clean",
                    min_max_report="min_max_report_preprocessing",
                ),
                name="remove_outliers",
            ),
        ],
        tags="pre_processing",
    )
