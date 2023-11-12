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

import types
import typing as tp

from itables import init_notebook_mode as _init_notebook_mode_for_tables

from ._init_for_plot_code import init_notebook_mode_for_code

_PLOT_CODE = "plot_code"
_INIT_TARGETS = types.MappingProxyType(
    {
        _PLOT_CODE: init_notebook_mode_for_code,
        "plot_table": _init_notebook_mode_for_tables,
    }
)
SUPPORTED_INIT_TARGETS = frozenset(_INIT_TARGETS.keys())
# interactive tables usually work without special activation
_DEFAULT_INIT_TARGETS = (_PLOT_CODE,)


def init_notebook_mode(target_init: tp.Iterable[str] = _DEFAULT_INIT_TARGETS) -> None:
    """
    Loads the js library and corresponding styles
    needed for fancy representation of plots.

    Warning: make sure you keep the output of this cell,
    otherwise plots won't work in the expected manner.

    Args:
        target_init: activates styles and js scripts for corresponding plots;
            currently supported targets: {'plot_code', 'plot_table'}
    """

    unknown_targets = SUPPORTED_INIT_TARGETS.difference(target_init)
    if unknown_targets:
        raise ValueError(
            f"Found unknown targets: {unknown_targets}. "
            f"Please use only following targets: {SUPPORTED_INIT_TARGETS}",
        )
    for target in set(target_init):
        init_function = _INIT_TARGETS[target]
        init_function()
