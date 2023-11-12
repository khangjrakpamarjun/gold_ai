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

TConfig = tp.TypeVar("TConfig")
TRawConfig = tp.Optional[tp.Dict[str, tp.Any]]


def parse_config(raw_config: TRawConfig, config_class: tp.Type[TConfig]) -> TConfig:
    return config_class(**raw_config) if raw_config else config_class()
