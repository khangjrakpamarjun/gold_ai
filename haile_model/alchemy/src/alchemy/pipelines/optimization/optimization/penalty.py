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

from typing import Any, Dict, List

import numpy as np

from optimizer import penalty
from optimizer.constraint import Penalty


class MassPullPenalty:
    def __init__(self, model_dict: Dict):
        self.model_dict = model_dict

    def __call__(self, x):
        mass_pull = self.model_dict["mass_pull"].predict(x)
        min_mp = 0.04  # TODO: make dynamic based on ore cluster
        max_mp = 0.10
        out = np.zeros_like(mass_pull)
        out = np.where(mass_pull > max_mp, mass_pull - max_mp, out)
        out = np.where(mass_pull < min_mp, min_mp - mass_pull, out)
        return out


def get_penalties(models_dict: Dict[str, Any]) -> Penalty:
    return penalty(
        MassPullPenalty(models_dict),
        "==",
        0,
        penalty_multiplier=0.5,
        penalty_function="linear",
        name="mass_pull",
    )
