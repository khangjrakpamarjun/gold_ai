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
Sequential feature selection
"""
# pylint:disable=too-many-statements, too-many-ancestors
import numbers
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_is_fitted

from ..base import Transformer


class CustomSequentialFeatureSelector(SequentialFeatureSelector, Transformer):
    """
    Modification of sklearn's Sequential Feature Selection
    to support:

        1. constrained selection of features (see features_to_keep)
           i.e. specifying a list of variables that must be included in
           the selected features.

        2. addition/removal of features until the scoring metric improves
           (see iter_till_improvement)

    Transformer that performs Sequential Feature Selection.
    This Sequential Feature Selector adds (forward selection) or
    removes (backward selection) features to form a feature subset in a
    greedy fashion. At each stage, this estimator chooses the best feature to
    add or remove based on the cross-validation score of an estimator.

    An example to illustrate constrained feature selection works:
    Let, n_features [i.e. len(X.columns)] = 9 , n_features_to_select = 5,
    n_features_to_keep [i.e. len(features_to_keep] = 3 ,
    iter_till_improvement = False

    Then, no. of features to select from = n_features - n_features_to_keep = 6,
    no. of features to add through feature
    selection = n_features_to_select - n_features_to_keep = 2 .

        * if direction = "forward" : select 2 features from 6 features by iterating
          twice, adding one feature in each iteration.
        * if direction = "backward" : select 2 features from 6 features by iterating
          four times, removing one feature in each iteration

    Attributes:
        n_features_to_select_ : The number of features that were selected.
        support_ : The mask of selected features.

    Examples::

       >>> from CustomSequentialFeatureSelector import
       ... CustomSequentialFeatureSelector
       >>> from sklearn.neighbors import KNeighborsClassifier
       >>> from sklearn.datasets import load_iris
       >>> X, y = load_iris(return_X_y=True)
       >>> X = pd.DataFrame(X)
       >>> X = X.rename({0: 'col0', 1: 'col1', 2: 'col2', 3: 'col3'}, axis=1)
       >>> knn = KNeighborsClassifier(n_neighbors=3)
       >>> sfs = CustomSequentialFeatureSelector(knn, n_features_to_select=3,
       ... features_to_keep=['col1', 'col3'])
       >>> sfs.fit(X, y)
       >>> sfs.get_support()
       array([False,  True,  True,  True])
       >>> sfs.transform(X).shape
       (150, 3)

    """

    def __init__(
        self,
        estimator,
        *,
        n_features_to_select=None,
        direction="forward",
        scoring=None,
        cv=5,
        n_jobs=None,
        iter_till_improvement=False,
        features_to_keep=None,
    ):
        """
        Args:
            features_to_keep : list , default= None
                List of features that cannot be discarded during feature selection. If
                if not None, for a given n_features_to_select, an additional
                n_features_to_select - len(features_to_keep) features will be selected.
            iter_till_improvement: bool, default=False
                If True, features are added (if direction = forward) or removed (if
                direction = backward) sequentially only if they improve the metric
                specified in scoring. Upon the first encounter with a feature that
                does not improve the scoring metric, further feature selection is
                terminated.

                * If True with direction = forward ,no. of features selected
                  <= n_features_to_select.
                * If True with direction = backward, no. of features selected
                  >= n_features_to_select.

            estimator : estimator instance
                An unfitted estimator.
            n_features_to_select : int or float, default=None
                The number of features to select. If `None`, n_features - 1 are
                selected. NOTE: THIS DIFFERS FROM default sklearn behaviour where
                if `None`, half the features are selected.If integer the parameter
                is the absolute number of features to select. If float between 0
                and 1, it is the fraction of features to select.
            direction : {'forward', 'backward'}, default='forward'
                Whether to perform forward selection or backward selection.
            scoring : str, callable, list/tuple or dict, default=None
                A single str (see :ref:`scoring_parameter`) or a callable
                (see :ref:`scoring`) to evaluate the predictions on the test set.
                NOTE that when using custom scorers, each scorer should return a single
                value. Metric functions returning a list/array of values can be wrapped
                into multiple scorers that return one value each.
                If None, the estimator's score method is used.
            cv : int, cross-validation generator or an iterable, default=None
                Determines the cross-validation splitting strategy.
                Possible inputs for cv are:
                * None, to use the default 5-fold cross validation,
                * integer, to specify the number of folds in a `(Stratified)KFold`,
                * :term:`CV splitter`,
                * An iterable yielding (train, test) splits as arrays of indices.
                For integer/None inputs, if the estimator is a classifier and ``y`` is
                either binary or multiclass, :class:`StratifiedKFold` is used. In all
                other cases, :class:`KFold` is used. These splitters are instantiated
                with `shuffle=False` so the splits will be the same across calls.
                Refer :ref:`User Guide <cross_validation>` for the various
                cross-validation strategies that can be used here.
            n_jobs : int, default=None
                Number of jobs to run in parallel. When evaluating a new feature to
                add or remove, the cross-validation procedure is parallel over the
                folds.
                ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
                ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
                for more details.
        Returns:
            self
        """
        # Additional kwargs
        self.features_to_keep = features_to_keep  # feature 1
        self.iter_till_improvement = iter_till_improvement  # feature 2

        super().__init__(
            estimator=estimator,
            n_features_to_select=n_features_to_select,
            direction=direction,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
        )

    # TODO: simplify: too much logic inside `fit` method
    def fit(self, X, y):  # noqa: WPS111,WPS231,WPS210,WPS238,N803
        """
        Learn the features to select.

        Args:
            X : array-like of shape (n_samples, n_features)
                If features_to_keep is not None, X must be
                pd.DataFrame
                Training vectors.
            y : array-like of shape (n_samples,)
                Target values.
        Returns:
            self : object
        """
        n_features = X.shape[1]

        # Addition 1 : feature 1
        # If self.features_to_keep is not None, get
        # indices of features_to_keep

        features_to_keep_idx = []
        n_features_to_keep = 0
        if self.features_to_keep is not None:

            # To enable column selection by column name, X must be of type pd.DataFrame
            if not isinstance(X, pd.DataFrame):
                type_of_input = type(X)
                raise TypeError(
                    "If features_to_keep is not None, "
                    "X must be of type pd.DataFrame",
                    f"Got {type_of_input}.",
                )

            n_features_to_keep = len(self.features_to_keep)

            # features_to_keep must be columns in X
            if set(self.features_to_keep) - set(X.columns):
                missing_columns = set(self.features_to_keep) - set(X.columns)
                raise ValueError(
                    "features_to_keep must be columns in X. "
                    f"Columns {missing_columns} not found.",
                )

            # for stable feature selection:
            # Needed because with
            X = X[sorted(X.columns)]  # noqa: WPS111,N806
            # getting indices of features_to_keep
            features_to_keep_idx = X.columns.get_indexer(self.features_to_keep).tolist()
        ###################################

        tags = self._get_tags()
        X, y = self._validate_data(  # noqa: WPS111,N806
            X,
            y,
            accept_sparse="csc",
            ensure_min_features=2,
            force_all_finite=not tags.get("allow_nan", True),
            multi_output=True,
        )

        error_msg = (
            "n_features_to_select must be either None, an "
            "integer in [1, n_features - 1] "
            "representing the absolute "
            "number of features, or a float in (0, 1] "
            "representing a percentage of features to "
            f"select. Got {self.n_features_to_select}"
        )

        # Addition 2: feature 1
        # Additional value errors if self.features_to_keep
        # is not None
        max_features_to_drop = n_features - n_features_to_keep
        if n_features_to_keep > 0:
            if self.n_features_to_select is not None:
                if self.n_features_to_select <= n_features_to_keep:
                    raise ValueError(
                        f"n_features_to_select should be > no. of features_to_keep."
                        f"Got {self.n_features_to_select}, {n_features_to_keep}",
                    )
            if max_features_to_drop < 2:
                raise ValueError(
                    "For feature selection to take place, the difference between t"
                    "he n_features "
                    "and no. of n_features_to_keep should be at least 2."
                    f"Got {max_features_to_drop}.",
                )
            if self.iter_till_improvement:
                warnings.warn(
                    "iter_till_improvement may over ride n_features_to_select",
                )
        ###################################

        if self.n_features_to_select is None:
            # Change 1 : feature 1
            # changed sklearn default behaviour to select n_features//2 features if
            # n_features_to_select is None to n_features -1.
            # This is to allow feature selection to take place if
            # n_features_to_keep > n_features//2
            self.n_features_to_select_ = n_features - 1  # noqa: WPS120
        elif isinstance(self.n_features_to_select, numbers.Integral):
            if not 0 < self.n_features_to_select < n_features:
                raise ValueError(error_msg)
            self.n_features_to_select_ = self.n_features_to_select  # noqa: WPS120
        elif isinstance(self.n_features_to_select, numbers.Real):
            if not 0 < self.n_features_to_select <= 1:
                raise ValueError(error_msg)
            self.n_features_to_select_ = int(  # noqa: WPS120
                n_features * self.n_features_to_select,
            )
        else:
            raise ValueError(error_msg)

        if self.direction not in {"forward", "backward"}:
            raise ValueError(
                "direction must be either 'forward' or 'backward'. "
                f"Got {self.direction}.",
            )

        cloned_estimator = clone(self.estimator)

        # the current mask corresponds to the set of features:
        # - that we have already *selected* if we do forward selection
        # - that we have already *excluded* if we do backward selection
        current_mask = np.zeros(shape=n_features, dtype=bool)

        # Addition 3 : feature 1
        # set current mask to True for features_to_keep for forward
        # selection to ensure that those features are already/always
        # selected
        current_mask[(features_to_keep_idx)] = self.direction == "forward"
        #####################################

        # Change 2 : feature 1
        n_iterations = self.n_features_to_select_ - n_features_to_keep
        # ex: 9 feats in all, 5 feats to select, of which 3 are mandatory and
        # direction = "forward"-> iter twice
        n_iterations = (
            n_iterations
            if self.direction == "forward"
            else n_features - self.n_features_to_select_
            # ex: 9 feats in all, 5 feats to select, of which 3 are mandatory and
            # direction = "backward"->
            # iter four times
        )

        ###################################

        # Addition 4: feature 2
        global_score = -np.inf
        ####################################

        for _ in range(n_iterations):
            # Change 3 : feature 2
            # _get_best_new_feature returns both new_feature_idx , curr_score
            new_feature_idx, curr_score = self._get_best_new_feature(
                cloned_estimator,
                X,
                y,
                current_mask,
                features_to_keep_idx,
            )
            ####################################

            # Addition 5 : feature 2
            # If iter_till_improvement == True, check if inclusion of feature
            # improves the relevant score/s
            # If not, end feature selection
            # If yes, include variable
            if self.iter_till_improvement:
                if curr_score > global_score:
                    global_score = curr_score
                else:
                    break
            ####################################
            current_mask[new_feature_idx] = True

        if self.direction == "backward":
            current_mask = ~current_mask
        self.support_ = current_mask  # noqa: WPS120

        return self

    def transform(self, x: pd.DataFrame):  # noqa: WPS111
        """Selects the features selected in `fit` from a provided dataframe."""
        check_is_fitted(self)
        self.check_x(x)
        return x.loc[:, self.support_]

    def _get_best_new_feature(
        self,
        estimator,
        X,  # noqa: WPS111,N803
        y,  # noqa: WPS111
        current_mask,
        features_to_keep_idx,
    ):
        # Return the best new feature to add to the current_mask, i.e. return
        # the best new feature to add (resp. remove) when doing forward
        # selection (resp. backward selection)
        candidate_feature_indices = np.flatnonzero(~current_mask)
        scores = {}
        # Loop through and add/ remove each feature during forward/backward selection
        # except the features that must be selected always
        for feature_idx in set(candidate_feature_indices) - set(features_to_keep_idx):
            candidate_mask = current_mask.copy()
            candidate_mask[feature_idx] = True
            if self.direction == "backward":
                candidate_mask = ~candidate_mask
            X_new = X[:, candidate_mask]  # noqa: N806
            scores[feature_idx] = cross_val_score(
                estimator,
                X_new,
                y,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
            ).mean()

        return (
            max(scores, key=lambda index: scores[index]),
            max(scores.values()),
        )
