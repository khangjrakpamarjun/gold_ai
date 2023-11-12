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

import pandas as pd
import pytz


def round_minutes(minute, interval):
    """Rounds time to  the nearest minute based on given
    time interval. Example 5:48:00 at 5min interval would
    return 5:50:00.

    Args:
        minute: minute component of current time
        interval: time interval used to stream data

    Returns:
        new minute based on interval
    """
    return (interval * round(minute / interval)) % 60


def get_current_time(interval, hourly=True):
    """Generate search start and end mostly for live
    streaming data using current time and streaming interval
    to determine search period. Default search is 1 day.

    Args:
        interval: time interval used to stream data
        hourly: boolean -> hourly stream or minute level stream

    Returns:
        search start and end time for data stream
    """
    cur_time = datetime.datetime.utcnow()
    if hourly:
        search_start = cur_time.replace(second=0, microsecond=0, minute=0)
    else:
        search_start = cur_time.replace(
            second=0,
            microsecond=0,
            minute=round_minutes(cur_time.minute, interval),
        )
    # get search time
    search_start = search_start.replace(tzinfo=None)
    search_end = search_start - datetime.timedelta(days=1)

    return search_start.isoformat(), search_end.isoformat()


def convert_timezone(pi_dataframe, timezone):
    """Converts data to timezone of the user

    Args:
        pi_dataframe: data streamed from pi api
        timezone: user's timezone for conversion. Default 'US/Eastern'
    """
    tz = pytz.timezone(timezone)  # 'US/Eastern'
    pi_dataframe.set_index("timestamp", inplace=True)

    pi_dataframe.index = pd.to_datetime(pi_dataframe.index).tz_convert(tz)
    pi_dataframe.index = pi_dataframe.index.tz_localize(None)
    return pi_dataframe.reset_index().rename(
        columns={pi_dataframe.index.name: "timestamp"},
    )
