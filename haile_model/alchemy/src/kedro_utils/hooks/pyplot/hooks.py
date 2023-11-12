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

from os.path import dirname, join, realpath
from typing import Any, Dict

import matplotlib.pyplot as plt
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from matplotlib.figure import Figure


class PyPlotStyleHooks(object):
    """
    Hooks for MatPlotLib style.
    """

    @hook_impl
    def before_pipeline_run(  # pylint:disable=unused-argument
        self,
        run_params: Dict[str, Any],
        pipeline: Pipeline,
        catalog: DataCatalog,
    ) -> None:
        """
        Hook will set filename to use for MatplotLib style based on the value of
        the 'mpl_theme' kedro run time parameter
        """

        # default to light theme
        style_name = run_params["extra_params"].get("mpl_theme", "light")
        path_to_style = join(dirname(__file__), f"{style_name}.mplstyle")
        plt.style.use(realpath(path_to_style))


class PyPlotCatalogHooks(object):
    """
    Hooks for MatplotLib figure annotations
    """

    @hook_impl
    def after_node_run(  # pylint:disable=unused-argument
        self,
        node: Node,
        catalog: DataCatalog,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        is_async: bool,
        run_id: str,
    ) -> None:  # pylint:disable=unused-argument
        """
        Hook will add an attribute 'catalog_entry` to all matplotlib figures that are
        returned as an output
        """

        for name, data in outputs.items():
            if isinstance(data, Figure):
                data.catalog_entry = name
