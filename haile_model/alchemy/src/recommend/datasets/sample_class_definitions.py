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


import pandas as pd
from sklearn.pipeline import Pipeline


class SilicaObjective(object):
    """Example objective class for tutorial using silica data."""

    def __init__(self, model: Pipeline):
        """Constructor.

        Args:
            model: object used to predict target.

        """
        self.model = model

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        """Call method to predict silica concentrate.
        Users can make a custom objective function suited for their usecases.
        Objective function can contain single model, multiple models, or an equation.
        """
        return self.model.predict(data)

    def __str__(self) -> str:
        return "Modeled silica concentrate objective"


class FlowPenalty(object):
    def __call__(self, data):
        return data["starch_flow"] + data["amina_flow"]


class OrePulpPhRepair(object):
    def __call__(self, data: pd.DataFrame):
        return data["ore_pulp_ph"]


def repair_column_by_setting_value(
    data: pd.DataFrame,
    column: str,
    value_to_set: float,
) -> pd.DataFrame:
    data = data.copy()
    data[column] = value_to_set
    return data
