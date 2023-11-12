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

import ipywidgets
import pandas as pd
from plotly import offline

from ._figures_manipulation import flatten_dict


@tp.runtime_checkable
class JupyterCompatible(tp.Protocol):
    """Represents protocol for objects with html repr in jupyter"""

    def _repr_html_(self) -> str:
        """Produces jupyter compatible html representation"""


_TFigureDictKey = tp.Union[str, int]
_TFigureDictValue = tp.Union[
    _TFigureDictKey,
    JupyterCompatible,
    tp.List[JupyterCompatible],
]
TFiguresDict = tp.Dict[_TFigureDictKey, tp.Union[_TFigureDictValue, "TFiguresDict"]]
TDictWithFloatsOrStr = tp.Dict[str, tp.Union[float, str]]


def create_plot_demonstration_widget(
    plot_set: TFiguresDict,
    sort_by_meta_data: tp.Optional[TDictWithFloatsOrStr] = None,
    nested_names_separator: str = ".",
    ascending: bool = True,
) -> None:
    """
    Creates widget from plot set

    Args:
        plot_set: figures to show, might be nested dict with figures
        nested_names_separator: if nested dict is provided, how to concat figure's
            prefix with name
        sort_by_meta_data: mapping from figure's name to figure's value to sort by
            in widget selector
        ascending: sorting order
    """
    offline.init_notebook_mode()
    # todo: move to notebook mode

    figures = flatten_dict(plot_set, nested_names_separator)
    order_of_keys_in_selector = (
        list(
            pd.Series(sort_by_meta_data).sort_values(ascending=ascending).index,
        )
        if sort_by_meta_data is not None
        else list(figures)
    )

    @ipywidgets.interact(plot_name=order_of_keys_in_selector)
    def show_figure(plot_name: str) -> JupyterCompatible:  # noqa: WPS430
        return figures[plot_name]
