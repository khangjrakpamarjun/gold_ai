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
Transformer of eng features
"""

from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
from pydantic.main import ModelMetaclass
from sklearn.utils.validation import check_is_fitted

from .pydantic_models import DerivedFeaturesCookBook
from .utils import Transformer


class NotDagError(Exception):
    """Raised when graph is expected to be a dag but is not"""


class FunctionReturnError(Exception):
    """Raise when a function to create an eng feature doesnt create it"""


class DAGHandlerMixin(object):
    def __init__(self, config: Dict[str, Any], pydantic_model: ModelMetaclass):
        """Mixin that handles parsing `config` according to `pydantic_model`,
        initializating `self._graph`, and providing a method to create a derived feature
        corresponding to a given node.

        Args:
            config (Dict[str, Any]): especification of eng features, constraints
             and models
            pydantic_model (ModelMetaclass): pydantic model used to parse `config`
        """
        self._graph = self.build_graph(pydantic_model(cookbook=config))
        self._config = config.copy()
        self._pydantic_model = pydantic_model

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @property
    def pydantic_model(self) -> ModelMetaclass:
        return self._pydantic_model

    @staticmethod
    def build_graph(cookbook: DerivedFeaturesCookBook) -> nx.DiGraph:
        graph = nx.DiGraph()
        for derived_feature, recipe in cookbook.cookbook.items():
            for dependency in recipe.dependencies:
                graph.add_edge(derived_feature, dependency)
            graph.nodes[derived_feature]["recipe"] = recipe
        if not nx.is_directed_acyclic_graph(graph):
            raise NotDagError("The graph of eng features is not a dag")
        return graph

    def _evaluate_node(self, data: pd.DataFrame, node_name: str) -> pd.DataFrame:
        recipe = self._graph.nodes[node_name]["recipe"]
        predictor = getattr(recipe.function, "predict", None)
        if predictor is not None:
            ans = recipe.function.predict(data)
        elif callable(recipe.function):
            ans = recipe.function(
                data,
                recipe.dependencies,
                *recipe.args,
                **recipe.kwargs,
            )
        else:
            function_name = recipe.function.__name__
            raise AttributeError(
                f"Recipe function {function_name}"
                " should be callable or have a predict method",
            )
        if isinstance(ans, (pd.Series, np.ndarray)):
            # TODO: Understand if using `assign` is the right way to avoid the warning
            #  Should work but does not: `data.loc[:, node_name] = ans`
            #  This despite the warning advising to use .loc
            data = data.assign(**{node_name: ans})
        else:
            data = ans
        return data


class FeatureFactory(Transformer, DAGHandlerMixin):
    def __init__(  # noqa: WPS612
        self,
        config: Dict[str, Any],
        pydantic_model: ModelMetaclass = DerivedFeaturesCookBook,
    ):
        """
        Creates the eng features especified by `config`. `config` must
        have the schema of `DerivedFeaturesCookBook`.

        Args:
            config (Dict[str, Any]): especification of eng features.

        Raises:
            NotDagError: if graph of eng features is not a dag.
        """
        super().__init__(config, pydantic_model)

    def fit(self, data: pd.DataFrame, target=None):
        """
        Checks the following:
        - if given input is a pandas DataFrame.
        - if leaf nodes of `graph` of eng features are present in `data`.
        - if functions associated with each eng feature actually create the feature.


        Args:
            data (pd.DataFrame): training data
            target: training target (no effect). Defaults to None.

        Raises:
            ValueError: if leaf node is not in `data`
            FunctionReturnError: if the function associated
             with an eng feature doesnt create it

        Returns:
            self
        """
        self.check_x(data)
        sample = data.head(2)
        filtered_toposort = []
        full_toposort = list(reversed(list(nx.topological_sort(self._graph))))

        for node_name in full_toposort:

            if self._not_leaf(node_name):
                filtered_toposort.append(node_name)
                before_derived = sample.copy()
                sample = self._evaluate_node(sample, node_name)
                self._validate_ans(sample, before_derived, node_name)

            elif node_name not in data:
                raise ValueError(
                    f"The following node must be present in x: {node_name}",
                )
        # sklearn trailing underscore naming convention is used here
        self.topological_sort_: List[str] = filtered_toposort  # noqa: WPS120
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates derived features according to `graph` cookbook

        Args:
            data (pd.DataFrame): data to create eng features

        Returns:
            pd.DataFrame: original data with eng features.
        """
        check_is_fitted(self, "topological_sort_")
        for node_name in self.topological_sort_:
            data = self._evaluate_node(data, node_name)
        return data

    def _not_leaf(self, node_name) -> bool:
        return self._graph.out_degree(node_name) > 0

    def _validate_ans(
        self,
        ans: pd.DataFrame,
        sample: pd.DataFrame,
        node_name: str,
    ) -> None:
        if not isinstance(ans, pd.DataFrame):
            raise FunctionReturnError(
                "The function of eng feature does not produce DataFrame",
            )
        columns_after_transform = set(ans.columns.tolist())
        expected_columns = set(sample.columns.tolist() + [node_name])
        if columns_after_transform != expected_columns:
            raise FunctionReturnError(
                f"The function of eng feature {node_name} does not create this feature",
            )


def create_features(params: Dict, data: pd.DataFrame) -> pd.DataFrame:
    """Wrapper function to run DerivedFeaturesMaker for Kedro pipeline

    Args:
        params (Dict): dict of kedro pipeline parameters
        data (pd.DataFrame): data

    Returns:
        pd.DataFrame: data with new features
    """

    transformer = FeatureFactory(params)
    transformer.fit(data)
    return transformer.transform(data)


def draw_graph(transformer: FeatureFactory, feature: Optional[str] = None):
    """This function plots maps DAGs used in DerivedFeaturesMaker.

    Args:
        transformer (DerivedFeaturesMaker): DerivedFeaturesMaker
        feature: If feature is provided function plots DAGs for specified feature
    """

    graph_options = {
        "node_color": "bisque",
        "node_size": 2000,
        "width": 3,
        "arrowsize": 12,
        "style": "--",
        "arrowstyle": "<|-",
        "font_size": 12,
        "font_color": "black",
    }
    feature_map = (
        transformer.config
        if feature is None
        else {feature: transformer.config[feature]}
    )
    graph = transformer.build_graph(transformer.pydantic_model(cookbook=feature_map))
    pos = nx.shell_layout(graph)
    nx.draw_networkx(graph, pos=pos, **graph_options)
