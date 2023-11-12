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
import typing as tp
from dataclasses import dataclass

import pandas as pd

import reporting.api.types as reporting_tp
from reporting.charts.primitive import plot_code, plot_table

from ._actual_vs_feature import plot_actual_vs_predicted, plot_actual_vs_residuals
from ._compare import plot_feature_comparison_for_train_test
from ._config_parser import TRawConfig, parse_config
from ._metrics_table import plot_train_test_metrics
from ._partial_dependence import plot_partial_dependency_for_sklearn
from ._shap import plot_shap_dependency, plot_shap_summary
from ._validation_approach import (
    plot_consecutive_validation_periods,
    plot_split_details,
    plot_validation_representation,
)
from .benchmark_models import fit_default_benchmark_models

logger = logging.getLogger(__name__)

_TBaselineModels = tp.Dict[str, reporting_tp.SupportsBaseModelAndEvaluateMetrics]

_NO_TITLE_LAYOUT_UPDATE = dict(title_text=None, margin_t=60)  # noqa: C408

_TABLES_WIDTH = 100
_SHAP_DEPENDENCY_SUBPLOT_WIDTH = 530
_SHAP_DEPENDENCY_SUBPLOT_HEIGHT = 400
_SHAP_DEPENDENCY_SPACING_PER_ROW = 0.52
_SHAP_DEPENDENCY_SPACING_PER_COLUMN = 0.4
_SHAP_SUMMARY_WIDTH = 800
_SKLEARN_PDP_PLOT_WIDTH = 12


@dataclass
class ModelPerformanceConfig(object):
    add_default_baselines: bool = True
    performance_table_sort_by: str = "mae"
    performance_table_sort_order: str = "asc"


@dataclass
class ValidationApproachConfig(object):
    sort_feature_comparison_by_shap: bool = True


@dataclass
class PDPSectionConfig(object):
    include: bool = True
    plot_individual_lines: bool = True
    random_state: int = 42
    n_columns: int = 2
    per_feature_height: float = 4
    n_jobs: int = -1
    drop_missing_values: bool = True


def get_modeling_overview(
    model: reporting_tp.Model,
    timestamp_column: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model_tuner: tp.Optional[reporting_tp.SupportsModelTunerBase] = None,
    model_factory: tp.Optional[reporting_tp.SupportsModelFactoryBase] = None,
    baseline_models: tp.Optional[_TBaselineModels] = None,
    model_performance_config: TRawConfig = None,
    pdp_section_config: TRawConfig = None,
    validation_approach_config: TRawConfig = None,
) -> reporting_tp.TFiguresDict:
    """
    Generates multi-level dict of figures for the Performance Report.

    There are 5 main sections in this report:
        * Model Introduction (model specification, features, target)
        * Validation Approach (visual of validation schema, features & target split
         comparisons)
        * Model Performance (model metrics and core visuals to access model quality)
        * Residual Analysis (deep dive in potential root causes of low performance)
        * Feature Importance (deep dive into main performance drivers)

    This report can be viewed in the notebook or
    used in report generation to produce standalone report file.

    Args:
        timestamp_column: column name of timestamp
        train_data: data containing train input features
        test_data: data containing test input features
        model: trained model
        model_tuner: tuner used to find optimal hyperparameters for model
        model_factory: model factory used to produce model
        model_performance_config: dict cast to `ModelPerformanceConfig` with attrs:
            performance_table_sort_by: metric used for sorting in model performance
             metrics
            performance_table_sort_order: sorting order
        baseline_models: mapping from names to models used for comparison with `model`;
            typically we want to have:
            * simple reference models here like AR1, previous month average,
            * reference models from previous iterations.
            By default, we provide AR1 model in addition to all baseline models passed.
        pdp_section_config: dict cast to `PDPSectionConfig` with attrs:
            * include: states whether this plot is included in results or not
            * plot_individual_lines: plots individual lines if true, only average
             otherwise
            * random_state: used for producing sampling
            * n_columns: columns in pdp subplots
            * per_feature_height: height of each plot
            * n_jobs: n processes to run when estimating; use -1 to activate max cores
        validation_approach_config: dict cast to `ValidationApproachConfig` with attrs:
            * sort_feature_comparison_by_shap: sorts train/test feature
                comparison in validation section by shap importance if true,
                leaves `features_in` order otherwise
    Returns:
        Dictionary of model performance figures
    """

    train_data = train_data.copy()
    test_data = test_data.copy()

    prediction_column = "__prediction"
    train_data[prediction_column] = model.predict(train_data)
    test_data[prediction_column] = model.predict(test_data)

    shap_explanation = model.produce_shap_explanation(train_data)

    # todo: add description for each section
    #  in performance maybe use plots numbering
    figs = {
        "Model Introduction": _get_introduction(model, model_tuner, model_factory),
        "Validation Approach": _get_validation_approach(
            model.target,
            model.features_in,
            model.get_shap_feature_importance_from_explanation(shap_explanation),
            timestamp_column,
            train_data,
            test_data,
            validation_config=parse_config(
                validation_approach_config,
                ValidationApproachConfig,
            ),
        ),
        "Model Performance": _get_model_performance(
            timestamp_column,
            prediction_column,
            train_data,
            test_data,
            model,
            baseline_models,
            config=parse_config(model_performance_config, ModelPerformanceConfig),
        ),
        "Residual Analysis": _get_residual_analysis(
            timestamp_column,
            prediction_column,
            train_data,
            test_data,
            model,
        ),
        "Feature Importance": _get_feature_importance(
            model,
            train_data,
            shap_explanation,
        ),
    }
    _add_pdp_section(
        figs=figs,
        data=train_data,
        model=model,
        plot_config=parse_config(pdp_section_config, PDPSectionConfig),
    )
    return figs


def _get_introduction(
    model: reporting_tp.SupportsBaseModel,
    model_tuner: tp.Optional[reporting_tp.SupportsModelTunerBase],
    model_factory: tp.Optional[reporting_tp.SupportsModelFactoryBase],
) -> reporting_tp.TFiguresDict:
    introduction = {
        "Model Definition": plot_code(repr(model), code_formatter="py-black"),
        "Target": plot_code(str(model.target)),
        "Features": plot_code(str("\n".join(model.features_in))),
    }
    if model_tuner is not None:
        introduction["Model Tuner"] = (
            plot_code(repr(model_tuner), code_formatter="py-black"),
        )
    if model_factory is not None:
        introduction["Model Factory"] = (
            plot_code(repr(model_factory), code_formatter="py-black"),
        )
    return introduction


def _get_validation_approach(
    target: str,
    features: tp.List[str],
    feature_importance: tp.Dict[str, float],
    timestamp_column: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    validation_config: ValidationApproachConfig,
) -> reporting_tp.TFiguresDict:
    """
    Returns: visual representation of validation which contains
        * Visual Representation
        * Consecutive Periods List
        * Feature on splits comparison
    """
    if validation_config.sort_feature_comparison_by_shap:
        features = sorted(
            features,
            key=lambda feature: feature_importance[feature],
            reverse=True,
        )
    return {
        "Visual Representation Of Split": [
            plot_split_details(train_data, test_data),
            plot_validation_representation(
                train_data,
                test_data,
                target,
                timestamp_column,
            ),
        ],
        "Consecutive Periods": plot_consecutive_validation_periods(
            train_data,
            test_data,
            timestamp_column,
        ),
        "Train vs. Test Comparisons": {
            "Target": plot_feature_comparison_for_train_test(
                train_data,
                test_data,
                [target],
            ),
            **plot_feature_comparison_for_train_test(
                train_data,
                test_data,
                features,
            ),
        },
    }


def _get_model_performance(
    timestamp_column: str,
    prediction_column: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model: reporting_tp.Model,
    baseline_models: tp.Optional[_TBaselineModels],
    config: ModelPerformanceConfig,
) -> reporting_tp.TFiguresDict:
    """Creates the figures for the overview of the model performance.

    These figures are
    - a table with the performance metrics
    - a graph of actual vs predicted (showing both train and test)
    - (if ``baseline_models`` are provided) a table comparing the performance metrics
     with those of the baseline models
    """
    if baseline_models is None:
        baseline_models = {}
    if config.add_default_baselines:
        benchmark_models = fit_default_benchmark_models(
            target=model.target,
            data=train_data,
            timestamp=timestamp_column,
        )
        baseline_models.update(benchmark_models)

    fig_metrics = plot_train_test_metrics(
        train_set_metrics=model.evaluate_metrics(train_data),
        test_set_metrics=model.evaluate_metrics(test_data),
        table_width=_TABLES_WIDTH,
    )
    fig_target_vs_predicted = plot_actual_vs_predicted(
        train_data=train_data,
        test_data=test_data,
        timestamp_column=timestamp_column,
        prediction_column=prediction_column,
        target_column=model.target,
    ).update_layout(
        _NO_TITLE_LAYOUT_UPDATE
    )  # removing duplicated title
    model_performance_figures = {
        "Metrics": fig_metrics,
        "Actual Target vs. Predicted": fig_target_vs_predicted,
    }
    if not baseline_models:
        return model_performance_figures
    fig_baselines = _get_baselines_comparison(
        model,
        baseline_models,
        test_data,
        metric_to_sort_by=config.performance_table_sort_by,
        sort_order=config.performance_table_sort_order,
    )
    model_performance_figures = {
        **model_performance_figures,
        "Baselines": fig_baselines,
    }
    return model_performance_figures  # noqa: WPS331  # Naming makes meaning clearer


def _get_baselines_comparison(
    model: reporting_tp.SupportsEvaluateMetrics,
    baseline_models: _TBaselineModels,
    test_data: pd.DataFrame,
    metric_to_sort_by: str,
    sort_order: str,
) -> reporting_tp.TFigure:
    performance_metrics = {
        model_name: model.evaluate_metrics(test_data)
        for model_name, model in baseline_models.items()
    }
    performance_metrics["Current Model"] = model.evaluate_metrics(test_data)
    df_performance_metrics = pd.DataFrame.from_dict(performance_metrics, orient="index")
    return plot_table(
        df_performance_metrics,
        width=_TABLES_WIDTH,
        sort_by=[(metric_to_sort_by, sort_order)],
    )


def _get_residual_analysis(
    timestamp_column: str,
    prediction_column: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model: reporting_tp.Model,
) -> reporting_tp.TFiguresDict:
    # todo: ? residuals by external category
    return {
        "Actual Target vs. Prediction Residuals": plot_actual_vs_residuals(
            train_data=train_data,
            test_data=test_data,
            timestamp_column=timestamp_column,
            prediction_column=prediction_column,
            target_column=model.target,
        ).update_layout(
            _NO_TITLE_LAYOUT_UPDATE
        ),  # removing duplicated title
    }


def _get_feature_importance(
    model: reporting_tp.Model,
    train_data: pd.DataFrame,
    shap_explanation: reporting_tp.ShapExplanation,
) -> reporting_tp.TFiguresDict:
    default_importance_column = "default_importance"
    shap_importance_column = "shap_importance"
    shap_feature_importance = model.get_shap_feature_importance_from_explanation(
        shap_explanation,
    )
    feature_importance_table = plot_table(
        data=pd.DataFrame(
            {
                default_importance_column: model.get_feature_importance(train_data),
                shap_importance_column: shap_feature_importance,
            },
        ),
        width=_TABLES_WIDTH,
        sort_by=[("shap_importance", "desc")],
        columns_to_color_as_bars=[
            default_importance_column,
            shap_importance_column,
        ],
    )
    shap_summary = (
        plot_shap_summary(
            model.features_in,
            shap_explanation,
            width=_SHAP_SUMMARY_WIDTH,
        )
        # removing duplicated title and increasing margin
        .update_layout(_NO_TITLE_LAYOUT_UPDATE)
    )
    shap_dependency_plot = plot_shap_dependency(
        model.features_in,
        shap_explanation,
        subplot_width=_SHAP_DEPENDENCY_SUBPLOT_WIDTH,
        subplot_height=_SHAP_DEPENDENCY_SUBPLOT_HEIGHT,
        horizontal_spacing_per_row=_SHAP_DEPENDENCY_SPACING_PER_ROW,
        vertical_spacing_per_column=_SHAP_DEPENDENCY_SPACING_PER_COLUMN,
    ).update_layout(_NO_TITLE_LAYOUT_UPDATE)
    return {
        "Feature Importance Table": feature_importance_table,
        "SHAP Summary": shap_summary,
        "SHAP Dependency Plot": shap_dependency_plot,
    }


def _add_pdp_section(
    figs: reporting_tp.TFiguresDict,
    data: pd.DataFrame,
    model: reporting_tp.Model,
    plot_config: PDPSectionConfig,
) -> None:
    if not plot_config.include:
        return
    try:
        pdp = plot_partial_dependency_for_sklearn(
            model=model,
            data=data,
            yaxis_title=model.target,
            title=None,
            plot_width=_SKLEARN_PDP_PLOT_WIDTH,
            plot_individual_lines=plot_config.plot_individual_lines,
            random_state=plot_config.random_state,
            n_columns=plot_config.n_columns,
            per_feature_height=plot_config.per_feature_height,
            n_jobs=plot_config.n_jobs,
            drop_missing_values=plot_config.drop_missing_values,
        )
    except AttributeError:
        logger.info("Partial dependency plots are not available for the model used.")
        return
    figs["Feature Importance"]["Partial Dependency Plot"] = pdp
