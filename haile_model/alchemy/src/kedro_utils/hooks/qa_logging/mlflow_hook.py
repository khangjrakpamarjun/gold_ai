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
from contextlib import suppress

with suppress(ImportError):
    import mlflow
    import mlflow.sklearn as mlf_sk
    import mlflow.tensorflow as mlf_tf

import os
import sys
import tempfile
from typing import Any, Dict

import kedro
import pandas as pd
import yaml
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from matplotlib.figure import Figure
from plotly import graph_objects as go


class MLFlowHooks(object):
    """Namespace for grouping all model-tracking hooks with MLflow together."""

    def __init__(self):
        # log models overwrites if more than one model are generated.
        # Also, we are logging the model dataset versions anyways.
        with suppress(NameError, ModuleNotFoundError):
            mlf_sk.autolog(log_models=False)
            mlf_tf.autolog(log_models=False)

    @kedro.framework.hooks.hook_impl
    def before_pipeline_run(
        self,  # pylint:disable=unused-argument
        run_params: Dict[str, Any],
        pipeline: Pipeline,
    ):  # pylint:disable=unused-argument
        """Hook implementation to set the storage location for mlflow data"""

        mlflow.start_run(run_name=run_params["run_id"])
        mlflow.log_params(run_params)

    @kedro.framework.hooks.hook_impl
    def after_node_run(  # noqa: WPS210, WPS231
        self,
        node: Node,
        outputs: Dict[str, Any],
        inputs: Dict[str, Any],  # pylint:disable=unused-argument
    ) -> None:  # pylint:disable=unused-argument
        """Hook implementation to add model tracking after some node runs.
        In this example, we will:

        * Log the parameters after the data splitting node runs.
        * Log the model after the model training node runs.
        * Log the model's metrics after the model evaluating node runs.

        Hook will add an attribute 'catalog_entry` to all matplotlib figures that are
        returned as an output
        """
        # Find plots if returned, or returned as dict of plots
        def _extract_figures(dict_obj, namespace_str=node.namespace):  # noqa: WPS430
            return {
                f"plots_{namespace_str}_{figure_name}": figure_object
                for figure_name, figure_object in dict_obj.items()
                if isinstance(figure_object, (Figure, go.Figure))
            }

        plot_artifacts = _extract_figures(outputs)

        for outputs_keys, outputs_values in outputs.items():
            # check if returning dict of figs as one output, then extract figures dict
            if isinstance(outputs_values, dict):
                plot_artifacts = {**plot_artifacts, **_extract_figures(outputs_values)}

            # Log model performance metrics
            if "metrics" in outputs_keys:

                metrics_dict = outputs[outputs_keys]
                # Log performance metrics -> convert to a dict if df is
                # returned with metric name on index
                if isinstance(outputs[outputs_keys], pd.DataFrame):
                    metrics_dict = dict(outputs[outputs_keys].to_records())

                metrics_dict = {
                    f"{node.short_name}.{metrics_keys}": metrics_values
                    for metrics_keys, metrics_values in metrics_dict.items()
                }
                mlflow.log_metrics(metrics_dict)

        if plot_artifacts:
            with tempfile.TemporaryDirectory() as temp_dir:
                for artifacts_keys, artifacts_values in plot_artifacts.items():
                    fig_path = os.path.join(temp_dir, f"{artifacts_keys}")
                    if isinstance(artifacts_values, Figure):
                        artifacts_values.savefig(fig_path)  # noqa: WPS220
                    elif isinstance(artifacts_values, go.Figure):
                        artifacts_values.write_html(f"{fig_path}.html")  # noqa: WPS220
                    mlflow.log_artifacts(temp_dir)

    @kedro.framework.hooks.hook_impl
    def after_pipeline_run(
        self,  # pylint:disable=unused-argument
        run_params: Dict[str, Any],
        run_result: Dict[str, Any],  # pylint:disable=unused-argument
        pipeline: Pipeline,
        catalog: DataCatalog,
    ) -> None:  # pylint:disable=unused-argument
        """Hook implementation to end the MLflow run
        after the Kedro pipeline finishes.
        """
        _log_kedro_info(run_params, pipeline, catalog)
        mlflow.end_run()


# pragma: no cover
def _log_kedro_info(  # noqa: WPS210
    run_params: Dict[str, Any],
    pipeline: Pipeline,
    catalog: DataCatalog,
) -> None:
    # this will have all the nested structures (duplicates)
    raw_params = {
        input_param: catalog._data_sets[input_param].load()  # noqa: WPS437
        for input_param in pipeline.inputs()
        if "param" in input_param
    }
    # similar to context.params
    raw_params.update(run_params.get("extra_params", {}))
    kedro_params = {}
    for param_name, param_value in raw_params.items():
        sanitised_parameter_name = _sanitise_kedro_param(param_name)
        kedro_params[f"kedro_{sanitised_parameter_name}"] = param_value
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_filename = f"{temp_dir}/params.yml"
        with open(temp_filename, "w") as temp_file:
            yaml.dump(kedro_params, temp_file, default_flow_style=False)
            mlflow.log_artifact(temp_filename)

    # Log tag dictionaries along with parameters once at end of pipeline run
    tag_dict_artifacts = {
        input_param: catalog._data_sets[input_param].load().to_frame()  # noqa: WPS437
        for input_param in pipeline.inputs()
        if "td" in input_param
    }
    with tempfile.TemporaryDirectory() as tag_dict_temp_dir:
        for td_name, td in tag_dict_artifacts.items():
            tag_dict_temp_filename = f"{tag_dict_temp_dir}/{td_name}.csv"
            td.to_csv(tag_dict_temp_filename, index=False)
            mlflow.log_artifact(tag_dict_temp_filename)
    mlflow.log_params(
        {
            "kedro_run_args": " ".join(
                repr(command_line_argument)
                if " " in command_line_argument
                else command_line_argument
                for command_line_argument in sys.argv[1:]
            ),
            "kedro_dataset_versions": list(_get_dataset_versions(catalog, pipeline)),
        },
    )


# pragma: no cover
def _sanitise_kedro_param(param_name):
    return param_name.replace(":", "_")


# pragma: no cover
def _get_dataset_versions(  # noqa: WPS231
    catalog: DataCatalog,
    pipeline: Pipeline,
):
    for ds_name, ds in sorted(catalog._data_sets.items()):  # noqa: WPS437
        ds_in_out = ds_name in pipeline.all_outputs()
        try:
            save_ver = ds.resolve_save_version() if ds_in_out else None
        except AttributeError:
            save_ver = None
        try:
            load_ver = ds.resolve_save_version() if ds_in_out else None
        except AttributeError:
            load_ver = None
        if save_ver or load_ver:
            version_info = {
                "name": ds_name,
                "save_version": save_ver,
                "load_version": load_ver,
            }
            yield version_info
