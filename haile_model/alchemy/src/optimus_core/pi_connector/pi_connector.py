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
import abc

import pandas as pd
import requests
from urllib3.exceptions import InsecureRequestWarning

# Suppress warnings
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
pd.options.mode.chained_assignment = None


class BaseOAIConnector(abc.ABC):
    def __init__(
        self,
        pi_endpoint,
        pi_api_user,
        pi_api_password,
        timezone="UTC",
        api_certificate=False,
    ):
        """Constructor

        Args:
            pi_endpoint: api endpoint of  osi pi system
            pi_api_user: api client username
            pi_api_password: api cleint password
            timezone: timezone to retrieve data in. Default is utc
            api_certificate: optional certificate verification check. Default is False
        """
        self.pi_endpoint = pi_endpoint
        self.pi_api_user = pi_api_user
        self.pi_api_password = pi_api_password
        self.timezone = timezone
        self.api_certificate = api_certificate

        # intialize url to connect to pi web server
        self.init_url = f"https://{self.pi_endpoint}/piwebapi/"

    @property
    def plant_data_attrs(self):  # noqa: WPS210
        """Collect all plant information about sensors and dataservers
        a single api call and returns a data dictionary with key
        attributes about each sensor in the plant. This can also be
        used as the base for building a data dictionary.

        Returns:
            pandas dataframe with information about each sensor in plant
        """
        # finalize url endpoint
        url = f"{self.init_url}dataservers"
        response = requests.get(
            str(url),
            auth=(self.pi_api_user, self.pi_api_password),
            verify=self.api_certificate,
            timeout=3,
        )
        # jsonify data response from server
        json_data = response.json()

        # collect all plant attributes data
        all_pi_points_data = []
        for dataserver in json_data["Items"]:
            pi_points_url = f"{url}/{dataserver['WebId']}/points"
            points = requests.get(
                str(pi_points_url),
                auth=(self.pi_api_user, self.pi_api_password),
                verify=self.api_certificate,
                timeout=3,
            )
            # jsonify data response from server
            points_data = points.json()
            # collect needed tag information
            data_list = [
                list(field_in_data_points)
                for field_in_data_points in zip(
                    *[
                        [
                            item["WebId"],
                            item["Name"],
                            item["Descriptor"],
                            item["PointType"],
                            item["EngineeringUnits"],
                            item["Zero"],
                            item["Span"],
                            item["Links"]["Value"],
                        ]
                        for item in points_data["Items"]  # noqa: WPS110
                    ],
                )
            ]
            # json data to pandas dataframe
            server_metadata = pd.DataFrame.from_records(data_list).T
            # set column names
            server_metadata.columns = [
                "web_id",
                "name",
                "description",
                "data_type",
                "eng_units",
                "low_limit",
                "high_limit",
                "value_link",
            ]
            server_metadata["data_server_id"] = dataserver["WebId"]
            server_metadata["data_server_name"] = dataserver["Name"]
            all_pi_points_data.append(server_metadata)

        return pd.concat(all_pi_points_data)

    def get_web_id(self, tag_name):
        """Query the webid of a sensor either through an api call
        or lookup from plant data attrs property

        Args:
            tag_name: Name of sensor as appears in osisoft pi system

        Returns:
            webid: string of random numbers and letters used in api call
        """
        # Search by querying plant attrs data or api call
        # TODO: refactor multiline condition
        if self.plant_data_attrs[  # noqa: WPS337
            self.plant_data_attrs["name"] == tag_name
        ]["web_id"].empty:
            web_id = self._query_tag_webid(tag_name)
        else:
            web_id = self.plant_data_attrs[self.plant_data_attrs["name"] == tag_name][
                "web_id"
            ].tolist()[0]

        return web_id

    def get_pi_data(self, post_data):
        """Abstract method used to load data from
        osisoft pi dataservers. Options include summaries,
        recorded and latest.

        Args:
            post_data:  json data of the batch sensors to be posted to api
        """
        # finalize url endpoint
        url = f"{self.init_url}batch"
        headers = {"Content-type": "application/json", "Accept": "application/json"}
        response = requests.post(
            str(url),
            auth=(self.pi_api_user, self.pi_api_password),
            json=post_data,
            # Verify=True enabled to have certification
            # verification check to make connection secure
            verify=True,
            headers=headers,
            timeout=3,
        )

        return self._get_pi_data(response.json())

    def query_sensor_data(self, tag_list):
        """Query Pi APi with a list of sensors to
        retrieve a unified dataframe all sensors matching timed
        intervals

        Args:
            tag_list: list of sensor to query

        Returns:
            all_sensor_data: unified dataframe with all sensors at intervals
        """
        # get plant attrs to be manipulated for api call
        tag_dict = self.plant_data_attrs
        return self._query_sensor_data(tag_list, tag_dict=tag_dict)

    @abc.abstractmethod
    def _get_pi_data(self, response_data):
        """Abstract method used to process data from
        osisoft pi dataservers. Options include summaries,
        recorded and latest.

        Args:
            response_data:  json data of all sensors being streamed
        """

    @abc.abstractmethod
    def _query_sensor_data(self, tag_list, tag_dict):
        """Query Pi APi with a list of sensors to
        retrieve a unified dataframe all sensors matching timed
        intervals
        """

    def _query_tag_webid(self, tag_name):
        """Queries the webid of a sensor with API call. Used in
        scenarios where webid is not available in plant attrs property

        Args:
            tag_name: Name of sensor as appears in osisoft pi system

        Returns:
            webid: string of random numbers and letters used in api call

        """
        # TODO helper to escape pagination
        # finalize url endpoint
        url = f"{self.init_url}search/query?q=name:{tag_name}"
        response = requests.get(
            str(url),
            auth=(self.pi_api_user, self.pi_api_password),
            verify=self.api_certificate,
            timeout=3,
        )  # verify=False will disable the certificate verification check

        # jsonify data response from server
        pi_api_response = response.json()
        # retrieve web_id by key
        return pi_api_response["Items"][0]["WebId"]
