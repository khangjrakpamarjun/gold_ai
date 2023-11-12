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
This is a SCRIPT for updating some datasets' artefacts.
Use it when structure of the module changes and pickles stop working.
"""

import logging
import pickle
import typing as tp

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from reporting.datasets import (
    DATA_DIR,
    CustomModelWrapper,
    ShapValues,
    get_master_table,
)
from reporting.datasets.calculate_shap_feature_impoortances import get_shap_explanation

_DEFAULT_RANDOM_SEED = 42

logger = logging.getLogger(__name__)


def _dump_dataset(
    dataset: pd.DataFrame,
    directory: str,
    file_name: str,
    **kwargs: tp.Any,
) -> None:
    return dataset.to_csv(DATA_DIR / f"{directory}/{file_name}.csv", **kwargs)


def _set_pickle_data(data: tp.Any, directory: str, file_name: str) -> None:
    with open(DATA_DIR / f"{directory}/{file_name}.pkl", "wb") as fw:
        pickle.dump(data, fw)


def _create_datasets(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tp.Iterable[tp.Tuple[str, pd.DataFrame]]:
    return ("train_data", train_data), ("test_data", test_data)


def _create_model() -> tp.Tuple[pd.DataFrame, pd.DataFrame, CustomModelWrapper]:
    data = get_master_table()
    train_data, test_data = train_test_split(
        data,
        random_state=_DEFAULT_RANDOM_SEED,
        shuffle=False,
    )
    features = [
        "inp_quantity",
        "cu_content",
        "inp_avg_hardness",
        "on_off_mill_a",
        "mill_b_power",
        "mill_a_power",
        "mill_b_load",
        "mill_a_load",
        "dummy_feature_tag",
    ]
    model = CustomModelWrapper(
        imputer=SimpleImputer(),
        estimator=RandomForestRegressor(random_state=_DEFAULT_RANDOM_SEED),
        target="outp_quantity",
        features_in=features,
    ).fit(train_data)
    explanation = get_shap_explanation(model, train_data)
    model.shap_values = ShapValues(
        data=explanation.data,
        values=explanation.values,
        base_values=explanation.base_values,
        feature_names=explanation.feature_names,
    )
    return train_data, test_data, model


def main() -> None:
    train_data, test_data, model = _create_model()
    model_data_dir = "model_results_mock_data"
    for file_name, dataset in _create_datasets(train_data, test_data):
        _dump_dataset(dataset, model_data_dir, file_name, index=False)
    _set_pickle_data(model, directory=model_data_dir, file_name="model")


if __name__ == "__main__":
    main()
