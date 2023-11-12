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


class LoggerMixin(abc.ABC):
    """
    Logging interface.
    """

    @abc.abstractmethod
    def log(self, *args, **kwargs):
        """Log something!
        Loggers must simply have a log method.

        Args:
            *args: positional arguments to pass to a logger.
            **kwargs: any keyword arguments to pass to a logger.
        """
