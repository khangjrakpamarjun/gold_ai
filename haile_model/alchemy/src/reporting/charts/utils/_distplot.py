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

import datetime
import typing as tp

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

BIN_NUMBER_UPPER_BOUND_RELATIVE_TO_DATA_SIZE = 0.9
BIN_NUMBER_UPPER_BOUND_MIN = 150.0
BEGINNING_OF_UNIX_TIME = "1970-01-01"
TVector = tp.Union[np.ndarray, pd.Series]
TTimeIntervalRelatedScalar = tp.Union[pd.Timedelta, datetime.timedelta]
TTimeIntervalRelatedSearchBound = tp.Tuple[
    TTimeIntervalRelatedScalar,
    TTimeIntervalRelatedScalar,
]


def calculate_optimal_bin_width(
    data: TVector,
    search_bounds: tp.Optional[tp.Tuple[float, float]] = None,
    search_precision: int = 1000,
) -> float:
    """
    Calculates optimal histogram bin width using cross-validation estimated squared
    error

    ``search_bounds`` are the min and max size of the bin. Default is None, if not
        provided then they are estimated
    ``search_precision`` is the number of candidate bin widths to be generated and
        examined in order to find the ideal number

    Notes:
        This function searches for the optimal bin width within the search bounds.
        If ``search_bounds`` are not provided, they are estimated by using
         ``_determine_bin_width_search_bounds``.
        Once the search bounds are defined, a number ``search_precision`` of candidate
         bin widths is generated, by taking a uniform sampling between min and max of
         the search bounds.
    """
    data_is_time_type = _is_time_data(data)
    if data_is_time_type:  # Makes sure to deal with numbers, not dates
        data, search_bounds = _convert_time_inputs_to_float(data, search_bounds)
    if search_bounds is None:
        search_bounds = _determine_bin_width_search_bounds(data=data)
    if min(search_bounds) == max(search_bounds):  # Only one value possible
        optimal_bin_size = min(search_bounds)
        if data_is_time_type:
            optimal_bin_size = pd.to_timedelta(optimal_bin_size, unit="S")
        return optimal_bin_size

    windows = np.linspace(*search_bounds, num=search_precision)
    errs = [
        _calculate_approximation_err_given_bin_width(data, window) for window in windows
    ]
    optimal_bin_size = windows[np.argmin(errs)]
    if data_is_time_type:
        optimal_bin_size = pd.to_timedelta(optimal_bin_size, unit="S")
    return optimal_bin_size


def _is_time_data(data: TVector):
    """Detect if data contains time-related information.

    Notes:
        Recognizes datetime types.
        Implementation tested for the following types:
            - pandas datetime64
            - numpy array containing
                - ``np.datetime64``
                - ``pd.Timestamp``
                - ``datetime.date``
        Note that ``datetime.datetime`` is a subclass of ``datetime.date,`` so the
         former is also accepted See [here](https://stackoverflow.com/questions/16991948/detect-if-a-variable-is-a-datetime-object)
         for more info
    """  # noqa: E501
    timestamp_types_in_numpy_array = [np.datetime64, pd.Timestamp, datetime.date]
    if isinstance(data, pd.Series):
        if is_datetime64_any_dtype(data):
            return True
    elif isinstance(data, np.ndarray):
        is_numpy_time_data = any(
            np.issubdtype(data.dtype, time_type)
            for time_type in timestamp_types_in_numpy_array
        )
        if is_numpy_time_data:
            return True
    return False


def _determine_bin_width_search_bounds(data: TVector):
    """Determines the lower and upper bounds for the optimal bin width to use in a
     histogram of ``data``.

    Notes:
        This function assumes the input ``data`` are of numeric type.
        Search bounds are estimated by taking:
        - the upper bound slightly wider than the data range (some buffer is added so
            the extrema of the datasets are both included).
            This is a good upper bound because it corresponds to (more or less) one
            single bin for all the data (aka "one bin to rule them all" :-D ).
        - a width that would cover the range with either 150 bins, or with
            almost as many bins as are datapoints (technically 0.9 bins per datapoint),
            whichever the highest.
            This is a good lower bound because it corresponds to (on average,
            approximately) one bin for each point

        Handles the case of constant data by returning a bin width of 1
    """
    data_range = data.max() - data.min()
    if data_range == 0:
        return 1.0, 1.0
    one_united_bin_width = data_range + 1  # small addition for max width
    large_number_of_bins = min(
        BIN_NUMBER_UPPER_BOUND_MIN,
        data.size * BIN_NUMBER_UPPER_BOUND_RELATIVE_TO_DATA_SIZE,
    )  # don't want it to be too large
    large_number_of_bins_width = data_range / large_number_of_bins
    search_bounds = (large_number_of_bins_width, one_united_bin_width)
    return search_bounds  # noqa: WPS331  # Naming makes meaning clearer


def _convert_time_inputs_to_float(
    data: TVector,
    search_bounds: tp.Optional[TTimeIntervalRelatedSearchBound],
):
    """Converts time-related data into floats.

    Used for preparing data for the calculation of the ideal bin width
    ``data``is a series with timestamp data
    ``bounds`` for the data is either ``None`` or a timedelta
    The float values are to be interperted as seconds
    """

    data = pd.to_timedelta(
        pd.Series(data) - pd.Timestamp(BEGINNING_OF_UNIX_TIME),
    ).dt.total_seconds()
    if search_bounds is not None:
        search_bounds = tuple(
            pd.to_timedelta(bound).total_seconds() for bound in search_bounds
        )
    return data, search_bounds


def _calculate_approximation_err_given_bin_width(
    data: TVector,
    bin_width: float,
) -> float:
    """
    Implements the method of minimizing integrated mean squared error (with
    leave-one-out) cross validation to determine the ideal number of bin.

    More info
    - in the original paper https://digitalassets.lib.berkeley.edu/sdtr/ucb/text/34.pdf
    - on Wikipedia https://en.wikipedia.org/wiki/Histogram#Minimizing_cross-validation_estimated_squared_error
    """  # noqa: E501
    if bin_width < 0:
        raise ValueError("Please pick window >= 0")

    n_points = data.size
    first_term = 2 / ((n_points - 1) * bin_width)

    n_bins = int(np.ceil((data.max() - data.min()) / bin_width))
    bin_counts, _ = np.histogram(data, n_bins)
    squared_sums = (bin_counts**2).sum()

    second_term = (
        squared_sums * (n_points + 1) / (n_points**2 * (n_points - 1) * bin_width)
    )
    err = first_term - second_term
    return err  # noqa: WPS331  # Namings makes meaning clearer
