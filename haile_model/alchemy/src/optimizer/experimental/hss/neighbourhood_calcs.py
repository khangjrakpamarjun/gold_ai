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
Class implementing Historical Similarity Sphere (HSS) sphere

"""


import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class NeighbourhoodCalculator:
    """
    Implements Historical Similarity sphere.
    This class is used to calculate points that fall within the neighbourhood of
    a given point by:

        1) finding closest `n_historical_points` to the current point using
            Euclidean distance
        2) if `expand_to_granular` is enabled increases density of the points
            by appending records from granular data frame

    Constructor of this class automatically applies scaling on the
    given dataset using `StandardScaler`.

    In practice low number of neighbours is usually used (100, even down to 30)
    based on aggregated dataset. It is possible to expand the dataset by using more
    granular dataset, by passing `expand_to_granular=True` in `get_points` method
    and overriding `_select_granular_df` based on your use case. This method
    needs to extract granular points corresponding to the aggregated,
    selected by the algorithm
    """

    def __init__(
        self,
        historical_agg_df: pd.DataFrame,
        granular_df: pd.DataFrame,
        distance_dims: List[str],
    ):
        """Initialises neighbourhood calculator. It works in the following way:
         - initializes historical data features by scaling to unit variance
         - calculates euclidean distance to all points from the current one
             (at shift/aggregate level). The current point is a function argument.

         - selects closest N points
         - increases density of the current N points by adding points from
             non-aggregated data

        Arguments:
            historical_agg_df (pd.DataFrame): historical dataset with aggregated level
                data
            granular_df (pd.DataFrame): historical non-aggregated dataframe
            distance_dims (List[str]): dimensions that are used to calculate the
                distance
        """
        self._historical_agg_df = historical_agg_df
        self._granular_df = granular_df
        self._distance_dims = distance_dims
        self._scaler = StandardScaler()
        self._scaler.fit(historical_agg_df[distance_dims])
        historical_scaled = self._scaler.transform(historical_agg_df[distance_dims])
        self._historical_scaled_df = pd.DataFrame(
            historical_scaled, columns=distance_dims, index=historical_agg_df.index
        )

    def get_points(
        self,
        point: pd.DataFrame,
        n_historical: int = 20,
        expand_to_granular: bool = False,
    ) -> pd.DataFrame:
        """Finds (at most) the `n_historical` points in the aggregated historical
            dataset which are 'nearest' to the given point.

        The `expand_to_granular` parameter controls whether the selected rows
            should be represented instead by more granular measurements. If this is
            true, this method may return more than `n_historical` points, depending
            on the implementation in `_select_granular_df`.

        Arguments:
            point (pd.DataFrame): current point
            n_historical (int): number of points to consider in
                neighbourhood (default: {20})
            expand_to_granular (bool): if True, enrichs aggregated, historical
                dataset with additional points from granular dataset.

        Returns:
            pd.DataFrame: returns points that are closest to the `point`
        """
        assert point.shape[0] == 1

        point_scaled = self._scaler.transform(point[self._distance_dims])
        distance = self._historical_scaled_df - point_scaled[0]
        distances_df = self._historical_scaled_df.copy()
        distances_df["distance"] = pd.Series(
            np.linalg.norm(distance, axis=1), index=distances_df.index
        )
        top_points_index = (
            distances_df.sort_values("distance", ascending=True)
            .head(n_historical)
            .index
        )

        if expand_to_granular:
            return_dataset = self._expand_to_granular(top_points_index)
        else:
            return_dataset = self._historical_agg_df.loc[top_points_index].copy()
        return return_dataset

    def _select_granular_df(self, aggregated_df: pd.DataFrame) -> pd.DataFrame:
        """Selects records from a granular dataset that forms the aggregated_df
        after applying the aggregation procedure. For example, if rows of the
        aggregated_df are hourly, this adds minute-level measurements.

        This method is to be overridden to fit your use case.

        Arguments:
            aggregated_df (pd.DataFrame): pd.DataFrame holding points that fall
                within the sphere

        Returns:
            pd.DataFrame:  matched granular records
        """
        raise NotImplementedError(
            "You need to override _select_granular_df method to adjust it to your"
            " use case"
        )

    def _expand_to_granular(self, top_points_index: pd.Index) -> pd.DataFrame:

        """Expands shiftly top points with points from hourly dataset

        Arguments:
            top_points_index (pd.Index): index of top points in
                `self._historical_agg_data`

        Returns:
            pd.DataFrame: combined deduped datasetset with aggregated and
                granular values
        """
        aggregated_df = self._historical_agg_df.loc[top_points_index].copy()

        granular_extension_df = self._select_granular_df(aggregated_df)
        combined_df = pd.concat([aggregated_df, granular_extension_df], sort=True)
        combined_df = combined_df[aggregated_df.columns].drop_duplicates()
        return combined_df
