from kedro.pipeline import Pipeline, node

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
