from kedro.pipeline import Pipeline, node, pipeline

from alchemy.pipelines.data_engineering.pre_processing.pre_processing_nodes import (
    remove_outliers,
)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=remove_outliers,
                inputs=[
                    "df_merged_with_derived_features",
                    "td",
                    "params:remove_outliers_derived_features",
                ],
                outputs=dict(
                    df="df_merged_with_features_outlier_clean",
                    min_max_report="min_max_report_post_processing",
                ),
                name="remove_outliers_postprocessing",
            ),
        ],
        tags="post_processing",
    )
