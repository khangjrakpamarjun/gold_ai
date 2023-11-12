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
"""
Core nodes performing postprocessing.
"""
import uuid
from typing import Any, Dict, List

import pandas as pd
from pandas.api.types import is_datetime64_dtype

from optimizer.solvers import Solver
from optimus_core.tag_dict import TagDict

from .utils import get_on_features, get_possible_values


def prepare_predictions(scores: Dict[pd.Index, pd.DataFrame]) -> List[Dict[str, Any]]:
    """Creates a list of predictions in the format of the `predictions` endpoint.

    Args:
        scores: a dictionary of optimization results.

    Returns:
        An input to 'predictions' endpoint of cra_api.
    """

    predictions_for_cra_api = []
    for score_row in scores.values():
        output_recommendation = {
            "run_id": score_row["run_id"].values[0],
            "tag_id": "objective",
            "baseline": float(score_row.loc["curr", "objective"]),
            "optimized": float(score_row.loc["opt", "objective"]),
            "id": str(uuid.uuid4()),
        }
        predictions_for_cra_api.append(output_recommendation)
    return predictions_for_cra_api


def prepare_recommendations(
    scores: Dict[pd.Index, pd.DataFrame],
    td: TagDict,
    solver_dict: Dict[pd.Index, Solver],
    default_status: str = "Pending",
    on_control_only: bool = True,
) -> List[Dict[str, Any]]:
    """Creates a list of recommndations in the format of the `recommendations` endpoint.

    Args:
        scores: a dictionary of optimization results.
        td: tag dictionary.
        solver_dict: dictionary containing `Solver`.
        default_status: default status. Defaults to "Pending".
        on_control_only: if True, only export controls that are on. Defaults to True

    Returns:
        An input to 'recommendations' endpoint of cra_api.
    """

    controls = td.select("tag_type", "control")
    recommendation_for_cra_api = []
    for score_row, solver in zip(scores.values(), solver_dict):
        on_controls = get_on_features(
            current_value=score_row.loc[["curr"]],
            td=td,
            controls=controls,
        )
        controls_to_export = on_controls if on_control_only else controls
        possible_values = get_possible_values(solver, on_controls)
        for control in controls_to_export:
            recommendation_for_cra_api.append(
                {
                    "tag_id": control,
                    "run_id": score_row["run_id"].values[0],
                    "value": score_row.loc["opt", control],
                    "id": str(uuid.uuid4()),
                    "status": default_status,
                    "comment": "",
                    "possible_values": possible_values.get(control, []),
                },
            )
    return recommendation_for_cra_api


def prepare_runs(
    scores: Dict[pd.Index, pd.DataFrame],
    iso_format: str = "%Y-%m-%dT%H:%M:%SZ",  # noqa: WPS323
    timestamp_col: str = "timestamp",
) -> List[Dict[str, str]]:
    """Create a list of runs in the format of `runs` endpoint of cra_api.

    Args:
        scores: a dictionary of optimization results.
        iso_format: format for timestamp.
        timestamp_col: column name for timestamp.

    Returns:
        An input to 'runs' endpoint of cra_api.
    """
    runs_for_cra_api = []
    for score_row in scores.values():
        score_row_timestamp = score_row[timestamp_col]
        if not is_datetime64_dtype(score_row_timestamp):
            score_row_timestamp = score_row_timestamp.astype("datetime64")

        score_row[timestamp_col] = score_row_timestamp.dt.strftime(iso_format)

        timestamp = score_row.loc["curr", timestamp_col]
        run_id = score_row.loc["curr", "run_id"]
        runs_for_cra_api.append({"id": run_id, "timestamp": timestamp})
    return runs_for_cra_api


def prepare_tags(td: TagDict) -> List[Dict[str, str]]:
    """Creates a list of tags in the format of `tags` endpoint of cra_api.

    Args:
        td: tag dictionary.

    Returns:
        An input to 'tags' endpoint of cra_api.
    """

    tags_df: pd.DataFrame = td.to_frame()
    if "area" not in tags_df.columns:
        tags_df["area"] = None
    tags_df = tags_df[["tag", "name", "area", "unit"]]
    tags_df.rename(columns={"tag": "id", "name": "clear_name"}, inplace=True)
    return tags_df.to_dict(orient="records")


def prepare_states(
    scores: Dict[pd.Index, pd.DataFrame],
    ui_states: List[str],
) -> List[Dict[str, str]]:
    """Creates a list of states in the format of `states` endpoint of cra_api.

    Args:
        scores: a dictionary of optimization results.
        ui_states: a list of state variables to show on UI.

    Returns:
        An input to 'states' endpoint of cra_api.
    """

    states_for_sra_api = []
    for score_row in scores.values():
        for state in ui_states:
            states_for_sra_api.append(
                {
                    "id": str(uuid.uuid4()),
                    "run_id": score_row["run_id"].values[0],
                    "tag_id": state,
                    "value": float(score_row[state].values[0]),
                },
            )
    return states_for_sra_api
