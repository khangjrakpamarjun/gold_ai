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
Central Tag Management class
"""

import logging
import typing as tp

import pandas as pd

from .dependencies import DependencyGraph
from .validation import TagDictError, validate_td

logger = logging.getLogger(__name__)
TRange = tp.Optional[tp.Tuple[float, float]]


# TODO: Refactor TagDict to reduce number of methods
class TagDict(object):  # noqa: WPS214
    """
    Class to hold a data dictionary. Uses a dataframe underneath and takes care of
    QA and convenience methods.
    """

    def __init__(self, data: pd.DataFrame, validate: bool = True) -> None:
        """
        Creates new TagDict object from pandas dataframe

        Args:
            data: input dataframe
            validate: whether to validate the input dataframe. validate=False can
             lead to a dysfunctional TagDict but may be useful for testing
        """
        self._validate = validate
        self._data = validate_td(data) if self._validate else data

        self._update_dependency_graph()

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name} data:\n{self._data}"

    def __getitem__(self, tag: str) -> tp.Dict[str, tp.Any]:
        """Gets all information about a given tag"""
        tag_data = self._get_tags_data([tag]).iloc[0]
        return tag_data.to_dict()

    def __contains__(self, tag: str) -> bool:
        """Checks whether a given tag exists in tag dict data"""
        return tag in set(self._data["tag"])

    def name(self, tag: str) -> str:
        """
        Returns clear name for given tag if tag exists and clear name is provided,
        or else tag name
        """
        if tag not in self:
            return tag

        tag_data = self[tag]
        return tag_data["name"] or tag

    def to_frame(self) -> pd.DataFrame:
        """Returns a copy of underlying dataframe"""
        data = self._data.copy()
        if self._validate:
            data["on_off_dependencies"] = data["on_off_dependencies"].apply(", ".join)
        return data

    def dependencies(self, tag: str) -> tp.List[str]:
        """Get all on_off_dependencies of a given tag"""
        self._check_are_known([tag])
        return self._dep_graph.get_dependencies(tag)

    def dependents(self, tag: str) -> tp.List[str]:
        """Returns all tags that input tag depends on"""
        self._check_are_known([tag])
        self._check_is_on_off([tag])
        return self._dep_graph.get_dependents(tag)

    def add_tag(self, tag_row: tp.Union[dict, pd.DataFrame]) -> None:
        """
        Adds new tag row/s to the TagDict instance,
        only if and entry doesn't already exist.

        Args:
            tag_row: DataFrame or Series/dict-like object of tag row/s

        Raises:
            TagDictError if the supplied tag rows are incorrect
        """
        if not isinstance(tag_row, (dict, pd.DataFrame)):
            type_of_tag_row = type(tag_row)
            raise TagDictError(
                f"Must provide a valid DataFrame or "
                f"dict-like object for the tag row/s. Invalid "
                f"object of type {type_of_tag_row} provided",
            )
        # Skip tags if already present in the TagDict.
        tag_data = pd.DataFrame(data=tag_row)
        tag_data.set_index("tag", inplace=True)

        tags_already_present = set(tag_data.index).intersection(set(self._data["tag"]))
        if tags_already_present:
            logger.info(
                f"[{tags_already_present}] already present in the Tag "
                f"Dictionary. Skipping.",
            )
            tag_data.drop(list(tags_already_present), inplace=True)

        if not tag_data.empty:
            data = self.to_frame()
            data = pd.concat(
                objs=[data, tag_data.reset_index()],
                axis=0,
                ignore_index=True,
                sort=False,
            )
            self._data = validate_td(data) if self._validate else data

            self._update_dependency_graph()

    def select(  # noqa: WPS231
        self,
        filter_col: tp.Optional[str] = None,
        condition: tp.Optional[tp.Any] = None,
    ) -> tp.List[str]:
        """
        Retrieves all tags according to a given column and condition. If no filter_col
        or condition is given then all tags are returned.

        Args:
            filter_col: optional name of column to filter by
            condition: filter condition
                       if None: returns all tags where filter_col > 0
                       if value: returns all tags where filter_col == values
                       if callable: returns all tags where filter_col.apply(callable)
                       evaluates to True if filter_col is present, or
                       row.apply(callable) evaluates to True if filter_col
                       is not present

        Returns:
            list of tags
        """

        def _condition(x):  # noqa: WPS430,WPS111

            # handle case where we are given a callable condition
            if callable(condition):
                return condition(x)

            # if condition is not callable, we will assert equality
            if condition:
                return x == condition

            # check if x is iterable (ie a row) or not (ie a column)
            try:
                iter(x)
            except TypeError:
                # x is a column, check > 0
                return x > 0 if x else False

            # x is a row and no condition is given, so we return
            # everything (empty select)
            return True

        data = self._data

        if filter_col:

            if filter_col not in data.columns:
                raise KeyError(f"Column `{filter_col}` not found.")

            mask = data[filter_col].apply(_condition) > 0
        else:
            mask = data.apply(_condition, axis=1) > 0

        return list(data.loc[mask, "tag"])

    def select_group_by(
        self,
        group_by: str,
        select: str = "tag",
        filter_by: tp.Optional[tp.List[str]] = None,
    ) -> tp.Dict[str, tp.List[str]]:
        """
        Groups tag dict data by `group_by`
        and returns a list of concatenated values within a single group from `select`.

        Args:
            group_by: column name by which to group tags
            select: column name which we use for getting aggreagated values
            filter_by: subset of `select` column values
                to use for resulting aggregations
                (i.e. only those that are in `filter_groups` will be selected),
                by default: all values from select will be used

        Returns:
            mapping from `group_by` column values to list of `select` column values

        Examples:
            Imagine you want to collect all tags grouped by areas
            like this::

                {
                    "area a": ["tag_a"],
                     "area b": ["tag_b"],
                      "area c": ["tag c", "tag_e"]
                }

            This can be done by::

                >>> td = TagDict(
                >>>     pd.DataFrame(
                >>>         {
                >>>             "tag": ["tag_a", "tag_b", "tag c", "tag_d", "tag_e"],
                >>>             "area": ["area a", "area b", "area c", None, "area c"],
                >>>         }
                >>>     ),
                >>>     validate=False,
                >>> )
                >>> td.select_group_by(group_by="area", filter_by=["tag_a", "tag_b"])
                ... {
                        "area a": ["tag_a"],
                        "area b": ["tag_b"],
                        "area c": ["tag c", "tag_e"],
                    }

            Not if you want to have areas only for tags "tag_a" and "tag_b"
            you can use `filter_groups` argument.
            This will result in `{"area a": ["tag_a"], "area b": ["tag_b"]}`::

                >>> td = TagDict(
                >>>     pd.DataFrame(
                >>>         {
                >>>             "tag": ["tag_a", "tag_b", "tag c", "tag_d", "tag_e"],
                >>>             "area": ["area a", "area b", "area c", None, "area c"],
                >>>         }
                >>>     ),
                >>>     validate=False,
                >>> )
                >>> td.select_group_by(group_by="area", filter_groups=[])
                ... {
                        "area a": ["tag_a"],
                        "area b": ["tag_b"],
                        "area c": ["tag c", "tag_e"],
                    }
        """
        data = self._data
        if filter_by is not None:
            is_requested_tag = data[select].isin(set(filter_by))
            data = data[is_requested_tag]

        return data.groupby(group_by)[select].apply(list).to_dict()

    def get_tag_ranges(
        self,
        tags: tp.Optional[tp.Iterable[str]] = None,
    ) -> tp.Dict[str, TRange]:
        if tags is None:
            tags = list(self._data["tag"])
        self._check_are_known(tags)

        tags_data = self._get_tags_data(tags) if tags is not None else self._data
        return dict(zip(tags, zip(tags_data["range_min"], tags_data["range_max"])))

    def get_model_features(self, features_column_name: str) -> tp.List[str]:
        """
        Get a list of tag names used as features for models train and prediction.

        Args:
            features_column_name: Name of the column in
             TagDict for model features indicators

        Returns:
            List of the tag names
        """
        return self.select(features_column_name)

    def _repr_html_(self) -> str:  # noqa: WPS120
        class_name = self.__class__.__name__
        html_representation = self._data._repr_html_()  # noqa: WPS437
        return f"<b>{class_name}" f" data</b>:<tr>{html_representation}"

    def _update_dependency_graph(self) -> None:
        """Update dependency graph to reflect what is currently in the tag dict"""

        graph = DependencyGraph()
        if "on_off_dependencies" in self._data.columns:
            all_deps = self._data.set_index("tag")["on_off_dependencies"].dropna()
            for tag, on_off_dependencies in all_deps.items():
                for dep in on_off_dependencies:
                    graph.add_dependency(tag, dep)
        self._dep_graph = graph

    def _get_tags_data(self, tags: tp.Iterable[str]) -> pd.DataFrame:
        self._check_are_known(tags)
        return self._data.set_index("tag").loc[tags].reset_index()

    def _check_are_known(self, tags: tp.Iterable[str]) -> None:
        """
        Check if tag is known

        Raises:
            KeyError: if some tag is missing
        """
        missing_tags = set(tags) - set(self._data["tag"])
        if missing_tags:
            raise KeyError(
                f"Following tags were not found in tag dictionary: {missing_tags}",
            )

    def _check_is_on_off(self, tags: tp.Iterable[str]) -> None:
        """
        Check tag is of `on_off` type

        Raises:
            TagDictError: if tag is not of `on_off` type
        """

        tags_data = self._get_tags_data(tags)
        is_wrong_tag_type = tags_data["tag_type"] != "on_off"
        wrong_tag_type_tags = tags_data.loc[is_wrong_tag_type, "tag"].tolist()
        if wrong_tag_type_tags:
            raise TagDictError(
                f"Following tags are not labelled as `on_off` "
                f"tag_type in the tag dictionary: {wrong_tag_type_tags}",
            )
