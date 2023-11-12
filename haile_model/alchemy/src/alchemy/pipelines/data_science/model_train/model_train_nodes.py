##########################################################################################
#                       Model train nodes
##########################################################################################

import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def create_no_soft_sensors(data: pd.DataFrame, params: Dict, td: Dict) -> pd.DataFrame:
    return data


def check_train_test_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame, params: Dict
):
    if len(train_data) == 0:
        raise ValueError("Train data is empty")

    if len(test_data) == 0:
        raise ValueError("Test data is empty")

    return None
