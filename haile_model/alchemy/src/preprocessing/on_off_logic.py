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
Nodes of the On off Logic pipeline.
"""

import logging

import pandas as pd

from optimus_core.tag_dict import TagDict

logger = logging.getLogger(__name__)


def set_off_equipment_to_zero(
    data: pd.DataFrame,
    td: TagDict,
) -> pd.DataFrame:
    """
    Mark sensor tags to zero based on the on/off tag dependencies defined
    in the Tag Dictionary.

    Args:
        data: input data
        td: Tag dictionary
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"data should be a Pandas dataframe, got {type(data)}",  # noqa: WPS237
        )

    tags = set(data.columns)

    on_off_tags = set(td.select("tag_type", "on_off")).intersection(tags)
    if not on_off_tags:
        logger.warning(
            "There are no on/off tags defined in Tag Dictionary "
            "which match any of the columns in the supplied dataframe",
        )
    tag_to_dependents = {
        on_off_tag: set(td.dependents(on_off_tag)).intersection(tags)
        for on_off_tag in on_off_tags
    }

    data = data.copy()

    # in cases where on-off tags have missing values, we impute with
    # the last known value. To change this behavior, consider implementing
    # custom logic here or earlier in the pipeline.
    data[list(on_off_tags)] = data[list(on_off_tags)].fillna(method="ffill")

    for on_off_tag, dependents in tag_to_dependents.items():
        if not dependents:
            continue

        # set tags to 0 when on/off tag is off. Change rule here as required.
        # For example could set to None or np.NaN instead.
        logger.info(
            f"Setting '{dependents}' to zero when '{on_off_tag}' is off.",
        )
        data.loc[data[on_off_tag] == 0, list(dependents)] = 0

    return data
