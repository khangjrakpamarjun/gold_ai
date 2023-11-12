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


import pandas as pd
from importlib_resources import files

DATA_DIR = files("feature_factory.data")


def get_sample_preprocessed_data() -> pd.DataFrame:
    """Example sample preprocessed data"""
    return pd.read_csv(DATA_DIR / "sample_preprocessed_data.csv")
