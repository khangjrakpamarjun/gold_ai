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

import warnings

from ... import api
from ..model_base import ModelBase


def verify_selected_controls(model: ModelBase, td: api.SupportsTagDict) -> None:
    """
    Log a message if a ModelBase instance has controls
    in the `features_in` (list of columns required for model training or inference),

    Args:
        model: trained instance of ModelBase
        td: tag dictionary.
    """
    controls = set(td.select("tag_type", "control"))
    model_controls_in = list(controls.intersection(model.features_in))
    if not model_controls_in:
        warnings.warn(
            "No controls found across model input features.",
            category=RuntimeWarning,
        )
