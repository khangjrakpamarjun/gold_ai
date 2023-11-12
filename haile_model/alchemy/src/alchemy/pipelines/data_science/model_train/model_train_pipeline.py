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

from alchemy.pipelines.data_science.model_train.model_train_nodes import (
    check_train_test_data,
    create_no_soft_sensors,
)
from modeling import (
    calculate_feature_importance,
    calculate_metrics,
    calculate_model_predictions,
    calculate_shap_feature_importance,
    create_model,
    create_model_factory_from_tag_dict,
    create_tuner,
    train_model,
    tune_model,
)


def create_train_model_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                check_train_test_data,
                inputs={
                    "train_data": "train_data",
                    "test_data": "test_data",
                    "params": "params:train_model",
                },
                outputs=None,
            ),
            node(
                create_model_factory_from_tag_dict,
                inputs={
                    "model_factory_type": "params:train_model.factory_class_name",
                    "model_init_config": "params:train_model.init",
                    "tag_dict": "td",
                    "target": "params:train_model.target_column",
                    "tag_dict_features_column": "params:train_model.td_features_column",
                },
                outputs="model_factory",
            ),
            node(
                create_model,
                inputs={"model_factory": "model_factory"},
                outputs="model",
            ),
            node(
                create_tuner,
                inputs={
                    "model_factory": "model_factory",
                    "model_tuner_type": "params:train_model.tune.class_name",
                    "tuner_config": "params:train_model.tune.tuner",
                },
                outputs="model_tuner",
            ),
            node(
                tune_model,
                inputs={
                    "model_tuner": "model_tuner",
                    "hyperparameters_config": "params:train_model.tune.hyperparameters",
                    "data": "train_data",
                },
                outputs="tuned_model",
            ),
            node(
                train_model,
                inputs={
                    "model": "tuned_model",
                    "data": "train_data",
                },
                outputs="trained_model",
                name="train_model",
            ),
            node(
                calculate_model_predictions,
                inputs={
                    "data": "train_data",
                    "model": "trained_model",
                },
                outputs="train_data_predictions",
                name="train_predict",
            ),
            node(
                calculate_model_predictions,
                inputs={
                    "data": "test_data",
                    "model": "trained_model",
                },
                outputs="test_data_predictions",
                name="test_predict",
            ),
            node(
                calculate_metrics,
                inputs={
                    "data": "train_data",
                    "model": "trained_model",
                },
                outputs="train_metrics",
                name="create_train_metrics",
            ),
            node(
                calculate_metrics,
                inputs={
                    "data": "test_data",
                    "model": "trained_model",
                },
                outputs="test_metrics",
                name="create_test_metrics",
            ),
            node(
                calculate_feature_importance,
                inputs={
                    "data": "train_data",
                    "model": "trained_model",
                },
                outputs="feature_importance",
                name="feature_importance",
            ),
            node(
                calculate_shap_feature_importance,
                inputs={
                    "data": "train_data",
                    "shap_producer": "trained_model",
                },
                outputs="shap_feature_importance",
                name="shap_feature_importance",
            ),
        ],
    ).tag("model_training")
