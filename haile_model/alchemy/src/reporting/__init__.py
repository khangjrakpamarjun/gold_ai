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
Reporting package contains following submodules:
    * ``api`` - provides users with an external API (currently contains only ``types``
     submodules)
    * ``charts`` - provides different plots / dashboards / overviews (dictionary of
     charts)
    * ``report`` - composes dictionary of charts into standalone sharable report file
    * ``interactive`` - provides widgets to ease charts wrangling in jupyter
    * ``datasets`` - contains mock data for showcasing functionality
    * ``testing`` - contains functions for testing purposes
"""

__version__ = "0.22.1"

from . import api, charts, datasets, interactive, report
