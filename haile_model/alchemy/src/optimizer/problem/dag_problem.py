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
Holds definitions for the DAGOptimizationProblem.
"""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Set, Union

import pandas as pd
from toposort import toposort

from optimizer.constraint.penalty import Penalty
from optimizer.constraint.repair import Repair
from optimizer.problem.stateful_problem import StatefulOptimizationProblem
from optimizer.types import Matrix, Predictor, Vector

_NODE_UPDATE_METHOD_INSERT = "insert"
_NODE_UPDATE_METHOD_OVERWRITE = "overwrite"

_VALID_METHODS = [_NODE_UPDATE_METHOD_INSERT, _NODE_UPDATE_METHOD_OVERWRITE]


class ComputationNode:
    """Class for representing an operation on a DataFrame in ComputationGraph."""

    def __init__(
        self,
        func: Union[Callable[..., Vector], Predictor],
        inputs: Union[List[str], str],
        output: str,
        update_method: str = _NODE_UPDATE_METHOD_INSERT,
        func_kwargs: Dict[str, Any] = None,
    ):
        """Constructor.

        Args:
            func: function called at the given node. First argument must accepted must
            be a pd.DataFrame.
            inputs: list of names or name of input columns to func.
            output: name of output column.
            update_method: how to update the output column.
            func_kwargs: keyword arguments to func.

        Raises:
            ValueError: if an invalid update method provided.
        """
        self._func = func
        self.inputs = [inputs] if isinstance(inputs, str) else inputs
        self.output = output
        self.func_kwargs = func_kwargs or {}

        if update_method not in _VALID_METHODS:
            raise ValueError(
                f"Given update method {update_method} is invalid. "
                f"Must be one of: {_VALID_METHODS}"
            )

        self.update_method = update_method

    @property
    def func(self) -> Callable[..., Vector]:
        """Expose the internal function or the ``predict``

        Returns:
            Callable.
        """
        if hasattr(self._func, "predict"):
            return self._func.predict

        return self._func

    def __repr__(self) -> str:
        """Repr method.

        Returns:
            str.
        """
        if hasattr(self._func, "__name__"):
            name = self._func.__name__

        else:
            name = repr(self._func)

            if "functools.partial" in name:
                name = "<partial>"

        return (
            f"ComputationNode(func={name}, inputs="
            f"{self.inputs}, output='{self.output}')"
        )

    def __str__(self) -> str:
        """String method.

        Returns:
            str.
        """
        return repr(self)

    def __lt__(self, other: Any) -> bool:
        """Less than method for comparing with strings.
        Necessary to get sorted lists in cyclic check.

        Args:
            other: Any.

        Returns:
            bool.
        """
        return str(self) < other

    def __gt__(self, other: Any) -> bool:
        """Greater than method for comparing with strings.
        Necessary to get sorted lists in cyclic check.

        Args:
            other: Any.

        Returns:
            bool.
        """
        return str(self) > other

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute node's function on a given DataFrame.

        Args:
            data: values to compute func with.

        Raises:
            KeyError: if the output column exists and the update method is
            not "overwrite".

        Returns:
            The updated DataFrame with new values.
        """
        if (
            self.output in data.columns
            and self.update_method != _NODE_UPDATE_METHOD_OVERWRITE
        ):
            raise KeyError(
                f"Column '{self.output}' exists in given data and update "
                f"method is '{self.update_method}'. Assign output of given function to "
                f"a new column or use update method '{_NODE_UPDATE_METHOD_OVERWRITE}'."
            )

        # Index ``data`` before passing to ensure users only access specified columns.
        data[self.output] = self.func(data[self.inputs], **self.func_kwargs)
        return data


class ComputationGraph:
    """Object to hold graph for computing updates in DAGOptimizationProblem."""

    def __init__(self, graph_list: List[Union[Dict[str, Any], ComputationNode]]):
        """Constructor.

        Args:
            graph_list: list of ComputationNodes are keyword argument dictionaries to
            pass to the ComputationNode constructor.
        """
        self.graph = list(toposort(self.build_graph(graph_list)))
        self.check_single_output()

    def check_single_output(self):
        """Check that no two nodes output the same value at the same step in topological
        order.

        Raises:
            ValueError if two nodes output the same value at the same step.
        """
        duplicates = []

        for nodes in self.graph:
            outputs, step_duplicates = set(), set()

            for node in nodes:
                if isinstance(node, ComputationNode):
                    if node.output in outputs:
                        step_duplicates.add(node)
                    outputs.add(node.output)

            if step_duplicates:
                duplicates += list(step_duplicates)

        if duplicates:
            node_strings = "\n".join(map(str, duplicates))
            raise ValueError(
                f"The following groups of computation nodes output the "
                f"same value in a topologically ordered step (and therefore their "
                f"computation order is nondeterministic):\n{node_strings}"
            )

    @staticmethod
    def build_graph(
        graph_list: List[Union[Dict[str, Any], ComputationNode]]
    ) -> Dict[Union[str, ComputationNode], Set[Union[str, ComputationNode]]]:
        """Build an adjacency list representation of the graph.

        Keys are nodes, values are dependencies (incoming edges).

        Args:
            graph_list: list of dictionaries describing the nodes to build.

        Returns:
            Adjacency list representing the given graph.
        """
        graph = defaultdict(set)

        for node in graph_list:
            if not isinstance(node, ComputationNode):
                node = ComputationNode(**node)

            for node_input in node.inputs:
                graph[node].add(node_input)

            graph[node.output].add(node)

        return graph

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Walk over the ComputationNodes and do their computations.

        Args:
            data: DataFrame to update and use for computations.

        Returns:
            Updated DataFrame.
        """
        for step in self.graph:
            for node in step:
                if isinstance(node, ComputationNode):
                    data = node.compute(data)

        return data


class DAGOptimizationProblem(StatefulOptimizationProblem):
    """OptimizationProblem that includes a computation graph for precomputing values
    to be used by repairs, penalties, and the objective. Also has all the functionality
    of a StatefulOptimizationProblem.

    Unlike the StatefulOptimizationProblem, DAGOptimizationProblems must use pandas
    DataFrames.
    """

    def __init__(
        self,
        objective: Callable[[Matrix], Vector],
        state: Union[pd.DataFrame, pd.Series],
        optimizable_columns: Union[List, pd.Series],
        penalties: Union[
            Penalty,
            Callable[[Matrix], Vector],
            List[Union[Penalty, Callable[[Matrix], Vector]]],
        ] = None,
        repairs: Union[
            Repair,
            Callable[[Matrix], Matrix],
            List[Union[Repair, Callable[[Matrix], Matrix]]],
        ] = None,
        graph: Union[
            ComputationGraph, List[Union[Dict[str, Any], ComputationNode]]
        ] = None,
        sense: str = "minimize",
    ):
        """Constructor.

        Args:
            objective: callable object representing the objective function.
            state: Vector representing the current state of the problem.
            optimizable_columns: list of columns/dimensions that can be optimized.
            penalties: optional Penalty or list of Penalties.
            repairs: optional Repairs or callable or list of Repairs and/or callables.
            graph: graph definition. Passed to ComputationGraph.
            sense: 'minimize' or 'maximize', how to optimize the objective.
        """
        super().__init__(
            objective,
            state,
            optimizable_columns,
            penalties=penalties,
            repairs=repairs,
            sense=sense,
        )

        self.graph = (
            graph
            if isinstance(graph, ComputationGraph)
            else ComputationGraph(graph or [])
        )

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute the internal graph's operations on ``data``.

        Args:
            data: DataFrame to pass to self.graph.compute.

        Returns:
            Updated DataFrame.
        """
        return self.graph.compute(data)

    def substitute_parameters(self, parameters: pd.DataFrame) -> pd.DataFrame:
        """Override the substitute parameters method to also compute the internal graph
        operations and update/add columns values.

        Args:
            parameters: DataFrame of parameters values to substitute current values into
            and pass to self.compute.

        Returns:
            DataFrame with update columns.
        """
        return self.compute(super().substitute_parameters(parameters))
