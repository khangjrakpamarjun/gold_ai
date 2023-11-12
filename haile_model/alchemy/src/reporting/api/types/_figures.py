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

from __future__ import annotations

import typing as tp
from abc import ABC, ABCMeta, abstractmethod

from reporting.interactive import JupyterCompatible
from reporting.report import HtmlCompatible, ImageCompatible


class ReprImplementationError(ValueError):
    """Is raised when implementation of any method is invalid and cannot be run"""


class PlotlyLike(JupyterCompatible, HtmlCompatible, tp.Protocol):
    """
    Protocol for plotly-like figures.
    We use two parent interfaces to show their properties:
    jupyter and html compatibility.

    See Also:
        `JupyterCompatible`
        `HtmlCompatible`
    """


class MatplotlibLike(JupyterCompatible, ImageCompatible, tp.Protocol):
    """
    Protocol for matplotlib-like figures.
    We use two parent interfaces to show their properties:
    jupyter and image compatibility.

    See Also:
        `JupyterCompatible`
        `ImageCompatible`
    """


class _FigureBaseMeta(ABCMeta):
    def __call__(cls, *args: tp.Any, **kwargs: tp.Any) -> FigureBase:
        instance = super().__call__(*args, **kwargs)
        _validate_representations(instance)
        return instance


def _validate_representation_to_html(instance: FigureBase) -> None:
    try:
        instance.to_html()
    except Exception as exc:
        raise ReprImplementationError(
            "Impossible to run `.to_html()` method",
        ) from exc


def _validate_representation_repr_html_(instance: FigureBase) -> None:
    try:
        instance._repr_html_()  # noqa: WPS437  # Ok because it is being tested
    except Exception as exc:
        raise ReprImplementationError(
            "Impossible to run `._repr_html_()` method",
        ) from exc


def _validate_representations(instance: FigureBase) -> None:
    _validate_representation_to_html(instance)
    _validate_representation_repr_html_(instance)


class FigureBase(ABC, metaclass=_FigureBaseMeta):
    """Implements `PlotlyLike` (preferable) protocol"""

    @abstractmethod
    def to_html(self) -> str:
        """Used for rendering in HTML report"""

    @abstractmethod
    def _repr_html_(self) -> str:
        """Used for rendering in Jupyter"""


TFigureDictKey = tp.Union[str, int]
TFigure = tp.Union[MatplotlibLike, PlotlyLike]
TFigureDictValue = tp.Union[TFigureDictKey, TFigure, tp.List[TFigure]]
TFiguresDict = tp.Dict[TFigureDictKey, tp.Union[TFigureDictValue, "TFiguresDict"]]

API_COMPATIBLE_TYPES = (PlotlyLike, MatplotlibLike)
