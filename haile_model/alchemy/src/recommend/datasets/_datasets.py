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


import pickle
import typing as tp

import pandas as pd
from importlib_resources import files
from sklearn.pipeline import Pipeline

from optimus_core import TagDict

DATA_DIR = files("recommend.data")


def get_sample_recommend_input_data() -> pd.DataFrame:
    """Example sample input data for recommend package"""
    return pd.read_csv(DATA_DIR / "sample_recommend_input_data.csv")


def get_sample_recommend_results() -> tp.Any:
    """Example sample recommend results"""
    return _get_pickle_data("sample_recommend_results")


def get_sample_tag_dict() -> TagDict:
    """Example sample tag dictionary data"""
    return TagDict(
        pd.read_csv(DATA_DIR / "sample_tag_dict.csv"),
        validate=False,
    )


def get_sample_optimization_explainer_tag_dict() -> TagDict:
    """
    Example sample tag dictionary producing
    artefacts for optimization explainer tutorial notebook
    """
    return TagDict(
        pd.read_csv(DATA_DIR / "sample_tag_dict_for_explainer.csv"),
        validate=False,
    )


def get_sample_problem_dict() -> tp.Any:
    return _get_pickle_data("sample_problem_dict")


def get_sample_solver_dict() -> tp.Any:
    """Example sample solver dict"""
    return _get_pickle_data("sample_solver_dict")


def get_trained_model() -> Pipeline:
    """Example sample trained model to run optimization with"""
    return _get_pickle_data("sample_trained_model")


def _get_pickle_data(file_name: str) -> tp.Any:
    byte_data = (DATA_DIR / f"{file_name}.pkl").read_bytes()
    return pickle.loads(byte_data)
