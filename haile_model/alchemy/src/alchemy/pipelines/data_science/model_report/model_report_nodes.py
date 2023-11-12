##########################################################################################
#                       Model report nodes
##########################################################################################

import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def model_report_information(params: Dict):
    logger.info(params["model_report_name"])
    return None
