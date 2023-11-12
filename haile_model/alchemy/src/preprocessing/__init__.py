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
This is a boilerplate pipeline
"""

from .cleaning import (  # noqa: F401
    apply_outlier_remove_rule,
    apply_type,
    convert_bool,
    deduplicate_pandas,
    enforce_custom_schema,
    enforce_schema,
    remove_null_columns,
    remove_outlier,
    replace_inf_values,
    series_convert_bool,
    unify_timestamp_col_name,
)
from .imputing import (  # noqa: F401
    ModelBasedImputer,
    fit_numeric_imputer,
    interpolate_cols,
    transform_numeric_imputer,
)
from .on_off_logic import set_off_equipment_to_zero  # noqa: F401
from .resampling import (  # noqa: F401
    get_valid_agg_method,
    resample_data,
    resample_dataframe,
)
from .timezones import round_timestamps  # noqa: F401
from .utils import (  # noqa: F401
    count_outlier,
    count_outside_threshold,
    create_range_map,
    create_summary_table,
)

__version__ = "0.9.4"
