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

from alchemy.pipelines.data_science.model_input.model_input_nodes import (
    aggregate_data,
    filter_data_by_target,
    filter_data_by_timestamp,
)
from modeling import create_splitter, drop_nan_rows, split_data


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
            # TODO: Add functionality of nan rows drop if different model required nans to be dropped
            # node(
            #     drop_nan_rows,
            #     inputs={
            #         "data": "data_filtered_by_target",
            #         "td": "td",
            #         "td_features_column": f"params:model_input.td_features_column",
            #         "target_column": f"params:model_input.target_column",
            #     },
            #     outputs="data_dropna",
            #     name="drop_nan_rows",
            # ),
            node(
                create_splitter,
                inputs={
                    "split_method": "params:model_input.split.split_method",
                    "datetime_column": "params:model_input.split.datetime_column",
                    "split_datetime": "params:model_input.split.split_datetime",
                },
                outputs="splitter",
                name="create_splitter",
            ),
            node(
                split_data,
                inputs={
                    "data": "data_filtered_by_target",
                    "splitter": "splitter",
                },
                outputs=["train_data", "test_data"],
                name="split_data",
            ),
        ],
    ).tag("model_input")
