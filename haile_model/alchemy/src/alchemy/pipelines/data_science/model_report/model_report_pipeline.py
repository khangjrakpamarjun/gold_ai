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

import reporting
from alchemy.pipelines.data_science.model_report.model_report_nodes import (
    model_report_information,
)
from modeling import check_model_features
from optimus_core.utils import partial_wrapper


def create_model_report_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                model_report_information,
                inputs={"params": "params:report"},
                outputs=None,
                name="report_information",
            ),
            node(
                check_model_features,
                inputs={
                    "td": "td",
                    "td_features_column": "params:report.td_features_column",
                },
                outputs="_model_features",
                name="get_model_features",
            ),
            node(
                partial_wrapper(
                    reporting.charts.get_modeling_overview,
                    pdp_section_config={"include": False},
                ),
                inputs={
                    "model": "trained_model",
                    "timestamp_column": "params:report.datetime_column",
                    "train_data": "train_data",
                    "test_data": "test_data",
                },
                outputs="performance_figures",
                name="generate_performance_figures",
            ),
            node(
                reporting.report.generate_html_report,
                inputs={
                    "figures": "performance_figures",
                    "report_meta_data": "params:report.performance.report_meta_data",
                    "render_path": "params:report.performance.render_path",
                },
                outputs=None,
                name="generate_performance_html_report",
            ),
        ],
    ).tag(["modeling", "reporting"])
