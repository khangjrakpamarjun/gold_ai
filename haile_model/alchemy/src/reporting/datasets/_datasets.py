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

import pickle
import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from importlib_resources import files
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from reporting.api.types import ShapExplanation
from reporting.charts.modeling.benchmark_models import evaluate_regression_metrics

TVector = tp.Union[pd.Series, np.ndarray]
TMatrix = tp.Union[npt.NDArray["np.generic"], np.matrix, pd.DataFrame]
TCustomModelWrapper = tp.TypeVar("TCustomModelWrapper", bound="CustomModelWrapperBase")
TFeatureImportanceDict = tp.Dict[str, float]

DATA_DIR = files("reporting.data")
PREDICTION_COLUMN = "prediction"


def get_batch_meta_with_features() -> pd.DataFrame:
    """Sample dataset batch analytics. Batch meta with features for batch-level.

    Returns:
        A `pandas.DataFrame` with 100 rows (100 batches) and the following columns:
        `['reactor_start_time', 'reactor_end_time', 'filter_start_time',
        'filter_end_time', 'reactor', 'filter']`
    """
    return _get_dataset(
        directory="batch_mock_data",
        file_name="batch_meta_with_features",
        index_col=0,
    )


def get_sensor_data_batched_phased() -> pd.DataFrame:
    """Sample dataset batch analytics. Batch sensors data labeled by batches and phases.

    Returns:
         A `pandas.DataFrame` with 13304 rows and the following columns:
        `['batch_id', 'time_step', 'datetime', 'filter_infeed', 'filter_trough_lvl'
        , 'reactor_P', 'reactor_acid_total', 'reactor_agitator_speed', 'reactor_temp']`.
    """
    return _get_dataset(
        directory="batch_mock_data",
        file_name="sensor_data_batched_phased",
        parse_dates=["datetime"],
    )


def get_mill_data() -> pd.DataFrame:
    """Sample dataset with mining data. Mill data."""
    return _get_dataset(directory="mining_mock_data", file_name="mill_historic")


def get_throughput_data() -> pd.DataFrame:
    """Sample dataset with mining data. Input & output sensors."""
    return _get_dataset(directory="mining_mock_data", file_name="in_out_historic")


def get_master_table() -> pd.DataFrame:
    """Sample dataset with model results. Master data."""
    return _get_dataset(directory="model_results_mock_data", file_name="master_table")


def get_train_data() -> pd.DataFrame:
    """
    Sample dataset with model results. Data used for training model.
    """
    return _get_dataset(directory="model_results_mock_data", file_name="train_data")


def get_test_data() -> pd.DataFrame:
    """
    Sample dataset with model results. Data used for testing model.
    """
    return _get_dataset(directory="model_results_mock_data", file_name="test_data")


def get_feature_importance_data() -> pd.Series:
    """
    Sample dataset with trained model's feature importance

    Example::
        | feature           |   feature_importance |
        |:------------------|---------------------:|
        | inp_quantity      |           0.852555   |
        | cu_content        |           0.0110595  |
    """
    model = get_mock_trained_model()
    train_data = get_train_data()
    df = pd.Series(
        model.get_feature_importance(train_data),
        name="feature_importance",
    )
    df.index.name = "feature"
    return df


def get_train_data_with_predictions() -> pd.DataFrame:
    """
    Train dataset with predictions in "prediction"
    column inferred by model from `get_mock_trained_model`
    """
    train_data = get_train_data()
    model = get_mock_trained_model()
    train_data[PREDICTION_COLUMN] = model.predict(train_data)
    return train_data


def get_test_data_with_predictions() -> pd.DataFrame:
    """
    Test dataset with predictions in "prediction"
    column inferred by model from `get_mock_trained_model`
    """
    test_data = get_test_data()
    model = get_mock_trained_model()
    test_data[PREDICTION_COLUMN] = model.predict(test_data)
    return test_data


def get_mock_trained_model() -> CustomModelWrapper:
    """
    Sample dataset with model results.
    Trained model.
    """
    return _get_pickle_data(directory="model_results_mock_data", file_name="model")


def _get_dataset(directory: str, file_name: str, **kwargs: tp.Any) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{directory}/{file_name}.csv", **kwargs)


def _get_pickle_data(directory: str, file_name: str) -> tp.Any:
    byte_data = (DATA_DIR / f"{directory}/{file_name}.pkl").read_bytes()
    return pickle.loads(byte_data)


@dataclass
class ShapValues(object):
    data: np.ndarray
    # Using ``values`` to keep the same interface as shap.Explanations
    values: np.ndarray  # noqa: WPS110
    base_values: np.ndarray
    feature_names: np.ndarray


def check_columns_exist(
    data: pd.DataFrame,
    col: tp.Union[str, tp.Iterable[str]] = None,
) -> None:
    if isinstance(col, str):
        if col not in data.columns:
            raise ValueError(f"Column {col} is not included in the dataframe.")

    if isinstance(col, tp.Iterable):
        columns = set(col)
        if not columns.issubset(data.columns):
            not_included_cols = columns - set(data.columns)
            raise ValueError(
                "The following columns are missing"
                f" from the dataframe: {not_included_cols}.",
            )


def _check_input_len_equals_prediction_len(data: TMatrix, prediction: TVector) -> None:
    if len(data) != len(prediction):
        raise ValueError("Length of input data is not the same as prediction length.")


class CustomModelWrapperBase(ABC):
    """
    Abstract base class for the dummy model for demonstrating OAI reporting
     functionality

    In order to remove dependencies between ``reporting`` and ``modeling`` this abstract
     base class redefines the relevant methods from ``ModelBase``, instead of inheriting
     from ``modeling``.
    """

    def __init__(
        self,
        estimator: RandomForestRegressor,
        target: str,
        features_in: tp.Iterable[str],
    ) -> None:
        self._estimator = estimator
        self._features_in = list(features_in)
        self._target = target

    @property
    def features_out(self) -> tp.List[str]:
        return self.features_in

    @property
    def target(self) -> str:
        return self._target

    @property
    def estimator(self) -> RandomForestRegressor:
        return self._estimator

    @property
    def features_in(self) -> tp.List[str]:
        return self._features_in.copy()

    def fit(
        self: TCustomModelWrapper,
        data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> TCustomModelWrapper:
        check_columns_exist(data, col=self._features_in)
        return self._fit(data, **kwargs)

    def predict(self, data: pd.DataFrame, **kwargs: tp.Any) -> TVector:
        check_columns_exist(data, col=self._features_in)
        prediction = self._predict(data, **kwargs)
        _check_input_len_equals_prediction_len(data, prediction)
        return prediction

    @abstractmethod
    def _predict(self, data: pd.DataFrame, **kwargs: tp.Any) -> TVector:
        """
        An abstract method for model prediction logic implementation.

        Mocking the ``ModelBase`` class.

        Args:
            data: DataFrame to make a prediction
            **kwargs: Additional keyword arguments that might be passed for model
             prediction

        Returns:
            A Series or ndarray of model prediction
        """

    @abstractmethod
    def _fit(
        self: TCustomModelWrapper,
        data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> TCustomModelWrapper:
        """
        An abstract method for model training implementation.

        Mocking the ``ModelBase`` class.

         Args:
            data: DataFrame to train model on
            **kwargs: Additional keyword arguments that might be passed for model
             training

        Returns:
            A trained instance of BaseModel class
        """


# TODO: Remove this warning when the docstring when the base classes are defined in
#  modeling. WARNING: The dosctring describes the future situation where the base
#  classes are defined in `modeling`
class CustomModelWrapper(CustomModelWrapperBase):
    """
    Dummy model for demonstrating OAI reporting functionality

    In order to remove dependencies between ``reporting`` and ``modeling`` the relevant
     methods are redefined here and in the CustomModelWrapperBase instead of inheriting
     the following ABCs from ``modeling``:
    - ``ModelBase`` (inherited from ``CustomModelWrapperBase``)
    - ``EvaluatesMetrics``
    - ``ProducesShapFeatureImportance``
    """

    def __init__(
        self,
        imputer: SimpleImputer,
        estimator: RandomForestRegressor,
        target: str,
        features_in: tp.Iterable[str],
    ) -> None:
        super().__init__(estimator, target, features_in)
        self._imputer = imputer
        self.shap_values: tp.Optional[ShapExplanation] = None

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        imputed_data = self._imputer.transform(data[self._features_in])
        return pd.DataFrame(imputed_data, columns=self._features_in)

    def get_feature_importance(
        self,
        data: pd.DataFrame,
        **kwargs,
    ) -> tp.Dict[str, float]:
        feature_importance = self._estimator.feature_importances_
        return dict(zip(self.features_out, feature_importance))

    def __repr__(self) -> str:
        imputer_representation = repr(self._imputer)
        estimator_representation = repr(self._estimator)
        return (
            "CustomModelWrapper(\n"
            f"    imputer={imputer_representation},\n"
            f"    model={estimator_representation},\n"
            f"    target={self.target},\n"
            f"    features_in={self._features_in},\n"
            ")"
        )

    def evaluate_metrics(
        self,
        data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> tp.Mapping[str, float]:
        """
        Calculate standard set of regression metrics:
            * Mean absolute error
            * Rooted mean squared error
            * Mean squared error
            * Mean absolute percentage error
            * R^2` (coefficient of determination)
            * Explained variance

        Args:
            data: data to calculate metrics
            **kwargs: keyword arguments passed to `.predict` method

        Returns:
            Mapping from metric name into metric value
        """
        target = data[self.target]
        prediction = self.predict(data)
        return evaluate_regression_metrics(target, prediction)

    def produce_shap_explanation(
        self,
        data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> ShapExplanation:
        """
        Returns precomputed shap_values instance

        To avoid extra dependencies on shap library we mock shap values estimation
        i.e. this model receives shap values and then returns it here

        Args:
            data: omitted
            **kwargs: omitted

        Returns:
            `shap.Explanation` containing prediction base values and SHAP values
        """

        if self.shap_values is None:
            raise ValueError(
                "Shap values are unavailable. Please set `shap_values` attribute.",
            )
        return self.shap_values

    def get_shap_feature_importance(
        self,
        data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> tp.Dict[str, float]:
        explanation = self.produce_shap_explanation(data, **kwargs)
        shap_feature_importance = np.abs(explanation.values).mean(axis=0)
        return dict(zip(explanation.feature_names, shap_feature_importance))

    @staticmethod
    def get_shap_feature_importance_from_explanation(
        explanation: ShapExplanation,
    ) -> TFeatureImportanceDict:
        """
        Calculate feature importance from the provided SHAP explanation.
        Importance is calculated as a mean absolute shap value.

        Args:
            explanation: shap-explanation object

        Returns:
            Mapping from feature name into numeric feature importance
        """
        shap_feature_importance = np.abs(explanation.values).mean(axis=0)
        return dict(zip(explanation.feature_names, shap_feature_importance))

    def _fit(self, data: pd.DataFrame, **kwargs: tp.Any) -> CustomModelWrapper:
        transformed_features = pd.DataFrame(
            self._imputer.fit_transform(data[self.features_in]),
            columns=self.features_out,
        )
        self._estimator.fit(transformed_features, data[self._target])
        return self

    def _predict(
        self,
        data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> npt.NDArray["np.generic"]:
        transformed_data = self.transform(data)
        return self._estimator.predict(transformed_data)
