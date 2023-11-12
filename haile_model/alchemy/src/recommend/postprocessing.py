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
from typing import Dict, List

import pandas as pd

from optimus_core.tag_dict import TagDict


def generate_run_id(
    scores: Dict[pd.Index, pd.DataFrame],
) -> Dict[pd.Index, pd.DataFrame]:
    """Generate run_id on optimization results.

    Args:
        scores: optimization results from bulk_optimize.

    Returns:
        updated scores with run_id
    """
    for score_row in scores.values():
        score_row["run_id"] = str(uuid.uuid4())

    return scores


class SummaryTable(object):
    """Summary table for uplift report."""

    def __init__(
        self,
        td: TagDict,
    ):
        """Constructor

        Args:
            td (TagDict): tag dictionary
        """

        self.td = td
        self.controls = td.select("tag_type", "control")

    def make_summary_table(
        self,
        scores,
        ui_states: List[str],
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        summary_table = []
        for score_row in scores:
            uplift_dict = {
                "run_id": score_row["run_id"].values[0],
                "timestamp": score_row[timestamp_col].values[0],
                "state": self._get_state_dict(score_row, ui_states),
                "controls": self._get_control_dict(score_row),
                "outputs": self._get_output_dict(score_row),
                "penalties": self._get_penalty_dict(score_row),
                "slack": self._get_slack_dict(score_row),
            }
            summary_table.append(uplift_dict)
        return pd.DataFrame(summary_table)

    def _get_output_dict(self, row: pd.DataFrame):
        return {
            "pred_current": float(row.loc["curr", "objective"]),
            "pred_optimized": float(row.loc["opt", "objective"]),
        }

    def _get_control_dict(self, row: pd.DataFrame):
        return {
            ctrl: {
                "current": float(row.loc["curr", ctrl]),
                "suggested": float(row.loc["opt", ctrl]),
                "delta": float(row.loc["opt", ctrl] - row.loc["curr", ctrl]),
            }
            for ctrl in self.controls
        }

    def _get_state_dict(self, row: pd.DataFrame, ui_states: List[str]):
        return {state: float(row[state].values[0]) for state in ui_states}

    def _get_penalty_dict(self, row: pd.DataFrame):
        return {
            penalty_column: {
                "current": float(row.loc["curr", penalty_column]),
                "suggested": float(row.loc["opt", penalty_column]),
            }
            for penalty_column in row.columns
            if "_penalty" in penalty_column
        }

    def _get_slack_dict(self, row: pd.DataFrame):
        return {
            slack_column: {
                "current": float(row.loc["curr", slack_column]),
                "suggested": float(row.loc["opt", slack_column]),
            }
            for slack_column in row.columns
            if "_slack" in slack_column
        }


def prepare_for_uplift_plot(scores: Dict[pd.Index, pd.DataFrame]) -> pd.DataFrame:
    """Prepares optimization results to feed into uplift plot.

    Args:
        scores: optimization results from bulk_optimize.

    Returns:
        Input for uplift plot
    """
    return (
        pd.concat(scores, axis=0)
        .reset_index(names=["index", "type"])
        .replace({"curr": "actual", "opt": "optimized"})
    )
