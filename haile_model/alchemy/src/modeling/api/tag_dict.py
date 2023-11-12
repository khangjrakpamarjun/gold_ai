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


import typing as tp


class SupportsTagDict(tp.Protocol):
    """
    Implementation of `optimus_core.tag_dict.TagDict` satisfies this protocol.

    This is a Protocol of tag dictionary that defienes API for the following methods:
        * select
        * get_model_features
    """

    def select(
        self,
        filter_col: tp.Optional[str] = None,
        condition: tp.Optional[tp.Any] = None,
    ) -> tp.List[str]:
        """
        Retrieves all tags according to a given column and condition. If no filter_col
        or condition is given then all tags are returned.
        """

    def get_model_features(self, features_column_name: str) -> tp.List[str]:
        """
        Get a list of tag names used as features for models train and prediction.

        Args:
            features_column_name: Name of the column in
             TagDict for model features indicators

        Returns:
            List of the tag names
        """
