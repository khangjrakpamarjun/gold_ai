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
Logger base module.
"""

import abc
import logging
from typing import List, Union

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class OptionalDependencyError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        Exception.__init__(self, message)
        self.message = message


class FileLoggerBase(abc.ABC):
    """
    File-Logging interface and methods.
    """

    def __init__(self, log_path):
        self.log_path = log_path

        # Instantiate writer
        try:
            self.tf_writer = tf.summary.create_file_writer(log_path)
        except AttributeError as error:
            logger.warning("File-based logging requires installation of tensorflow")
            raise OptionalDependencyError(
                f"{error}. Please install tensorflow to use file-writing loggers."
            )

    def write_array_to_disk(
        self, data: Union[List, np.ndarray], iteration: int, name: str
    ):
        """
        Method to write self.data_dict to disk for tensorboard

        Args:
            data: iterable/array to write
            iteration: optimization iteration
            name: Name of data to save as (for plotting).
        """
        if self.log_path:
            with self.tf_writer.as_default():  # pylint: disable=not-context-manager
                tf.summary.histogram(name, data, step=iteration)
