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
We try to keep all subpackages/submodules less dependent on each other.
To do so each submodule might introduce its own protocols.
And when it does, we have to make sure that it's in our API
(if it's supposed to be used by other modules or users, of course).

Hence, this subpackage consolidates main extremal and internal types used in package.
Those types consist of model and figure types.

Model types are used to show the requirements from model side.
Like the fact that model has target and features attributes or that it can
produce shape feature importance.

We express those expectations in following protocols:
    * Model
    * SklearnModel
    * ModelFactory
    * ModelTuner
    * Estimator
    * ShapExplanation
    * SupportsBaseModel
    * SupportsModelFactoryBase
    * SupportsModelTunerBase
    * SupportsBaseModelAndEvaluateMetrics
    * SupportsEvaluateMetrics
    * SupportsShapFeatureImportance

Figure types are used to show that we not rely on exact libraries implementations.
Instead, we describe what is expected of figure and how one can implement his own
 figure.

We express those expectations in following protocols and types:
    * HtmlCompatible
    * ImageCompatible
    * JupyterCompatible
    * MatplotlibLike
    * PlotlyLike

We also provide `FigureBase` for building user defined figures.

There is a collections of those types that aggregate those protocols:
    * TFiguresDict
    * TFigureDictKey
    * TFigureDictValue
    * TFigure
    * API_COMPATIBLE_TYPES (tuple of figures united in `TFigure`)
TFiguresDict is a recursive dictionary of `TFigureDictKey` -> `TFigureDictValue`.
Where on each level we can either have `TFigure`/list of `TFigure` or nested
 `TFiguresDict`
"""


from ._figures import (
    API_COMPATIBLE_TYPES,
    FigureBase,
    HtmlCompatible,
    ImageCompatible,
    JupyterCompatible,
    MatplotlibLike,
    PlotlyLike,
    ReprImplementationError,
    TFigure,
    TFigureDictKey,
    TFigureDictValue,
    TFiguresDict,
)
from ._model_related import Estimator, ShapExplanation
from ._models import (
    Model,
    SklearnModel,
    SupportsBaseModel,
    SupportsBaseModelAndEvaluateMetrics,
    SupportsEvaluateMetrics,
    SupportsModelFactoryBase,
    SupportsModelTunerBase,
    SupportsShapFeatureImportance,
)
