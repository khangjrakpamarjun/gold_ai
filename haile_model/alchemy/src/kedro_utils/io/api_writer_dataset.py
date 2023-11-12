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
"""Api writer dataset."""

from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import requests
from kedro.extras.datasets.api import APIDataSet
from kedro.io.core import DataSetError
from requests.auth import AuthBase


class APIWriterDataSet(APIDataSet):
    """``APIWriterDataSet`` saves data to HTTP(S) APIs.
    It uses the python requests library: https://requests.readthedocs.io/en/master/

    Example:
    ::

        >>> from core_pipelines.kedro_utils.io import APIWriterDataSet
        >>>
        >>>
        >>> data_set = APIWriterDataSet(
        >>>     url=<my_url>,
        >>>     params={
        >>>         "key": "SOME_TOKEN",
        >>>     }
        >>> )
        >>> data_set.save({"key": "value"})
    """

    def __init__(
        self,
        url: str,
        method: str = "POST",
        params: Dict[str, Any] = None,
        headers: Dict[str, Any] = None,
        auth: Union[Tuple[str], AuthBase] = None,
        timeout: int = 60,
        iterative: bool = False,
    ) -> None:
        """Creates a new instance of ``APIDataSet`` to fetch data from an API endpoint.

        Args:
            url: The API URL endpoint.
            method: The Method of the request, GET, POST, PUT, DELETE, HEAD, etc...
            params: The url parameters of the API.
                https://requests.readthedocs.io/en/master/user/quickstart/#passing-parameters-in-urls
            headers: The HTTP headers.
                https://requests.readthedocs.io/en/master/user/quickstart/#custom-headers
            auth: Anything ``requests`` accepts. Normally it's either
                ``('login', 'password')``, or ``AuthBase``, ``HTTPBasicAuth`` instance
                for more complex cases.
            timeout: The wait time in seconds for a response, defaults to 1 minute.
                https://requests.readthedocs.io/en/master/user/quickstart/#timeouts
            iterative: whether or not to store data on one call or on several calls.

        """
        self._iterative = iterative
        super().__init__(
            url,
            method=method,
            params=params,
            headers=headers,
            auth=auth,
            timeout=timeout,
        )

    def _execute_request(  # pylint:disable=arguments-differ
        self,
        data: Any = None,
    ) -> requests.Response:
        self._request_args["json"] = data
        return super()._execute_request()

    def _save(self, data: Any) -> Union[List[requests.Response], requests.Response]:
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient="records")
        if isinstance(data, list) and self._iterative:
            for element in data:
                self._execute_request(element)
        else:
            self._execute_request(data)

    def _load(self) -> None:
        class_name = self.__class__.__name__
        raise DataSetError(f"{class_name} is a write only data set type")
