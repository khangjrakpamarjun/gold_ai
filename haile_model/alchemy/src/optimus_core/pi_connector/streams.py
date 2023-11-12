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
from functools import reduce

import pandas as pd

from .pi_connector import BaseOAIConnector
from .utils import convert_timezone, get_current_time


class OAIRecordedStreams(BaseOAIConnector):
    def __init__(
        self,
        pi_endpoint,
        pi_api_user,
        pi_api_password,
        start_time=None,
        end_time=None,
    ):
        """Constructor

        Args:
            pi_endpoint: api endpoint of  osi pi system
            pi_api_user: api client username
            pi_api_password: api cleint password
            start_time (default=None): start time of stream (iso format date)
            end_time (default=None): end time of stream (iso format date)

        """
        # intialize vars from base class
        super().__init__(pi_endpoint, pi_api_user, pi_api_password)

        self.start_time = start_time if start_time else get_current_time(1)[0]
        self.end_time = end_time if end_time else get_current_time(1)[1]

    def _get_pi_data(self, response_data):
        """Query pi servers for sensor data using parameters as inputs to return
        summary data at specified intervals with correct aggregations.
        Summary is based on aggregation with options available
        https://devdata.osisoft.com/piwebapi/help/topics/summary-type

        Args:
            response_data: json data of all sensors being streamed

        Returns:
            dataframe with all sensor data in the correct timezone
        """
        all_tag_values = []
        for tag, response_data_for_tag in response_data.items():
            timestamp = []
            data_point = []
            for j_object in response_data_for_tag["Content"]["Items"]:
                # collect timestamp
                timestamp.append(j_object["Timestamp"])
                # collect and handle dict value response for digital tags
                sensor_data_for_tag = (
                    j_object["Value"]["Value"]
                    if isinstance(j_object["Value"], dict)
                    else j_object["Value"]
                )
                data_point.append(sensor_data_for_tag)
            # json data to pandas dataframe
            sensor_data = pd.DataFrame.from_records([timestamp, data_point]).T
            # set column names
            sensor_data.columns = ["timestamp", tag]

            all_tag_values.append(convert_timezone(sensor_data, self.timezone))

        return all_tag_values

    def _query_sensor_data(self, tag_list, tag_dict):
        """Query Pi APi with a list of sensors to
        retrieve a unified dataframe all sensors matching timed
        intervals

        Args:
            tag_list: list of sensor to query. Use sensor name
                      as they appear in  osisoft pi system

        Returns:
            all_sensor_data: unified dataframe with all sensors at
                             specified intervals
        """
        # filter plant attrs for only sensors of interest
        query_dict = tag_dict[tag_dict["name"].isin(tag_list)]
        # tag url to use for data extraction per sensor
        query_dict["Resource"] = (
            self.init_url
            + "streams/"
            + query_dict["web_id"]
            + "/recorded?"
            + f"startTime={self.end_time}&"
            f"endTime={self.start_time}&"
            f"boundaryType=Inside&maxCount=150000"
        )
        # set api call method
        query_dict["Method"] = "GET"

        # dataframe to json as payload for api call
        query_dict.set_index("name", inplace=True)
        query_dict = query_dict[["Method", "Resource"]]
        post_data = query_dict.to_dict("index")

        # get sensor data
        return self.get_pi_data(post_data)


class OAISummaryStreams(BaseOAIConnector):
    def __init__(
        self,
        pi_endpoint,
        pi_api_user,
        pi_api_password,
        start_time=None,
        end_time=None,
        interval=1,
        hourly=True,
        aggregation="Average",
    ):
        """Constructor

        Args:
            pi_endpoint: api endpoint of  osi pi system
            pi_api_user: api client username
            pi_api_password: api cleint password
            start_time (default=None): start time of stream (iso format date)
            end_time (default=None): end time of stream (iso format date)
            interval: optional stream interval. Default is 1hour
            hourly: optional houlry stream. Default is True
            aggregation: option aggregation emthood. Default is Average
        """

        self.interval = interval
        self.aggregation = aggregation
        self.hourly = hourly
        # intialize vars from base class
        super().__init__(pi_endpoint, pi_api_user, pi_api_password)

        self.start_time = (
            start_time if start_time else get_current_time(self.interval)[0]
        )
        self.end_time = end_time if end_time else get_current_time(self.interval)[1]

    def _get_pi_data(self, response_data):
        """Query pi servers for sensor data using parameters as inputs to return
        interpolated data at specified intervals with correct aggregations
        """
        all_tag_values = []
        for tag, response_data_for_tag in response_data.items():
            timestamps = []
            data_point = []
            for j_object in response_data_for_tag["Content"]["Items"]:
                # collect value
                data_point.append(j_object["Value"]["Value"])
                # collect and handle dict timestamp response
                timestamp = (
                    j_object["Value"]["Timestamp"]
                    if "Type" in j_object
                    else j_object["Timestamp"]
                )
                timestamps.append(timestamp)

            # json data to pandas dataframe
            sensor_data = pd.DataFrame.from_records([timestamps, data_point]).T
            # set columns names
            sensor_data.columns = ["timestamp", tag]

            all_tag_values.append(sensor_data)

        # merge all sensor response into a single dataframe
        return reduce(
            lambda left, right: pd.merge(left, right, on=["timestamp"], how="inner"),
            all_tag_values,
        )

    def _query_sensor_data(self, tag_list, tag_dict):
        """Query Pi APi with a list of sensors to
        retrieve a unified dataframe all sensors matching timed
        intervals

        Args:
            tag_list: list of sensor to query. Use sensor name
                      as they appear in  osisoft pi system
        Returns:
            all_sensor_data: unified dataframe with all sensors at
                             specified intervals
        """
        # filter plant attrs for only sensors of interest
        query_dict = tag_dict[tag_dict["name"].isin(tag_list)]
        # set interval cadence if data is stream hourly or minute level
        if self.hourly:
            interval_cadence = "h"
        else:
            interval_cadence = "m"
        # interpolation url for digital tags
        interpolate = (
            self.init_url
            + "streams/"
            + query_dict["web_id"]
            + "/interpolated?"
            + f"startTime={self.end_time}&"
            f"endTime={self.start_time}&"
            f"interval={self.interval}{interval_cadence}&"
            f"boundaryType=Inside&maxCount=150000"
        )
        # summary url for float and int tags
        summary = (
            self.init_url
            + "streams/"
            + query_dict["web_id"]
            + "/summary?"
            + f"startTime={self.end_time}&"
            f"endTime={self.start_time}&"
            f"summaryType={self.aggregation}&calculationBasis=TimeWeighted&"
            f"timeType=MostRecentTime&"
            f"summaryDuration={self.interval}{interval_cadence}&maxCount=150000"
        )
        # tag url to use for data extraction per sensor
        query_dict["Resource"] = pd.np.where(
            query_dict.data_type == "Digital",
            interpolate,
            summary,
        )
        # set api call method
        query_dict["Method"] = "GET"

        # dataframe to json as payload for api call
        query_dict.set_index("name", inplace=True)
        query_dict = query_dict[["Method", "Resource"]]
        post_data = query_dict.to_dict("index")

        # get sensor data
        sensor_data = self.get_pi_data(post_data)
        return convert_timezone(sensor_data, self.timezone)


class OAICurrentValueStreams(BaseOAIConnector):
    def _get_pi_data(self, response_data):
        """Query pi servers for sensor data using parameters as inputs to return
        interpolated data at specified intervals with correct aggregations

        Args:
            tag_name: Name of sensor as appears in osisoft pi system

        Returns:
            dataframe with current sensor value in the correct timezone
        """
        current_tag_values = {}
        for tag, tag_specific_data in response_data.items():
            # collect current values for all tags
            current_tag_values[tag] = (
                tag_specific_data["Content"]["Value"]["Value"]
                if isinstance(tag_specific_data["Content"]["Value"], dict)
                else tag_specific_data["Content"]["Value"]
            )
        return current_tag_values

    def _query_sensor_data(self, tag_list, tag_dict):
        """Query Pi APi with a list of sensors to
        retrieve a unified dataframe all sensors matching timed
        intervals

        Args:
            tag_list: list of sensor to query. Use sensor name
                      as they appear in  osisoft pi system
            tag_dict: plant data attrs

        Returns:
            all_sensor_data: unified dataframe with all sensors at
                             specified intervals
        """
        # filter plant attrs for only sensors of interest
        query_dict = tag_dict[tag_dict["name"].isin(tag_list)]
        # set api call method
        query_dict["Method"] = "GET"
        # tag url to use for data extraction per sensor
        query_dict["Resource"] = query_dict["value_link"]

        # dataframe to json as payload for api call
        query_dict.set_index("name", inplace=True)
        query_dict = query_dict[["Method", "Resource"]]
        post_data = query_dict.to_dict("index")
        # get sensor data
        sensor_query = self.get_pi_data(post_data)
        sensor_data = pd.DataFrame([sensor_query], columns=sensor_query.keys())
        sensor_data["timestamp"] = pd.Timestamp.utcnow()

        return convert_timezone(sensor_data, self.timezone)
