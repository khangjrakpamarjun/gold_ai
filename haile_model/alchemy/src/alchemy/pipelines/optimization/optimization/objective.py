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
import logging
from typing import Dict

import pandas as pd

from modeling import ModelBase

logger = logging.getLogger(__name__)


class UpstreamObjective(object):
    def __init__(self, models_dict: Dict, parameters: Dict):
        """Constructor.

        Args:
            models_dict: Dictionary of models such as {"tph": tph.trained_model, 'rec':
            rec.trained_model}

        """
        self.models_dict = models_dict
        self.parameters = parameters

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        # Return objective
        w1 = self.parameters["weights"]["sag_power"]
        w2 = self.parameters["weights"]["sulphide_grade"]

        w1 = w1 / 10_000  # SAG power =~ 10_000 x sulphide_grade

        power = self.models_dict["sag_power"].predict(data)
        sulphide_grade = self.models_dict["sulphide_grade"].predict(data)

        return w1 * power - w2 * sulphide_grade


def make_objective(models_dict: Dict, parameters: Dict) -> UpstreamObjective:
    """Make a new ``SilicaObjective``.

    Args:
        models_dict: Dictionary of models such as {"tph": tph.trained_model, 'rec':
        rec.trained_model}
        parameters: Dictionary for objective function.

    Returns:
        UpstreamObjective instance.
    """
    parameters = parameters["obj_function"]

    return UpstreamObjective(models_dict=models_dict, parameters=parameters)
