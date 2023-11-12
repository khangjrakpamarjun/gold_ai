from kedro.pipeline import Pipeline, node, pipeline

from alchemy.pipelines.data_engineering.prepare_test_data.prepare_test_data_nodes import (
    prepare_test_data,
)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_test_data,
                inputs=[
                    "df_merged_with_features_outlier_clean",
                    "td",
                    "params:test_data_input",
                ],
                outputs="test_data",
                name="prepare_test_data",
            ),
        ],
        tags="test_data_prepare",
    )
