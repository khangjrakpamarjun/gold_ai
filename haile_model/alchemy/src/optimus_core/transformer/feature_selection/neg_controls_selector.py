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
"""Selecting class which selects based on column data types."""
# pylint: disable=attribute-defined-outside-init
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

from ..base import Transformer
from ..estimator import Estimator


class SimpleNegControlSelector(Transformer):
    """
    Selects features with higher importance than most important random feature.

    Attributes:
        support_: A numpy array/boolean mask of which columns are selected
        selected_features_: list of features that are selected.

    """

    def __init__(
        self,
        estimator: Optional[RegressorMixin] = None,
        n_random_features: Optional[int] = 5,
        noise_distribution: Optional[str] = "uniform",
        random_state: Optional[int] = 1234,
    ):
        """Instantiate the selector and attach parameters.

        Args:
            estimator: An instantiated estimator to fit to the data and retrieve
                feature importance values from.
            n_random_features: The number of random features to create.
                Default is 5.
            noise_distribution: The distribution to use to sample the random
            features, where it is an attribute of numpy random. Default is uniform.
            random_state: Random state for the random number generator.
        """
        if estimator is None:
            self.estimator = RandomForestRegressor()
        else:
            self.estimator = estimator
        self.estimator = estimator
        self.n_random_features = n_random_features
        self.noise_distribution = noise_distribution
        self.random_state = random_state

    def fit(
        self,
        x: pd.DataFrame,  # noqa: WPS111
        y: Optional[pd.DataFrame] = None,  # noqa: WPS111
        **fit_params,
    ) -> "SimpleNegControlSelector":
        """Fits the selector.

        Fits the selector by running an estimator on the given dataframe and additional
            random features, and selecting those features in the data that have a higher
            feature importance than the most important random feature.

        Args:
            x: Data containing the features for which we will compute the importance.
            y: The target values used to fit the estimator.
            **fit_params: Other estimator specific parameters.

        Returns:
            A fitted selector.

        """
        self.check_x(x)
        x_rnd = self.create_random_features(x)
        self.estimator.fit(x_rnd, y, **fit_params)

        feat_importances = _get_feature_importances(self.estimator)

        random_feat_importances = feat_importances[
            -self.n_random_features :  # noqa: E203
        ]
        feat_importances_fte = feat_importances[: -self.n_random_features]

        max_rnd_imp = random_feat_importances.max()

        mask = np.ones_like(feat_importances_fte, dtype=bool)
        mask[feat_importances_fte < max_rnd_imp] = False

        self.support_ = mask
        self.selected_features_ = x.columns[self.support_].tolist()
        return self

    def get_support(self) -> np.ndarray:
        """Provides a boolean mask of the selected features."""
        check_is_fitted(self)
        return self.support_

    def transform(self, x: pd.DataFrame):  # noqa: WPS111
        """Selects the features selected in `fit` from a provided dataframe."""
        check_is_fitted(self)
        self.check_x(x)
        return x.loc[:, self.support_]

    def create_random_features(
        self,
        x: pd.DataFrame,  # noqa: WPS111
    ):
        """Samples random features and adds them to the input dataframe."""
        n_sample = len(x)

        rnd_ftes = np.random.RandomState(  # noqa: WPS609
            self.random_state,
        ).__getattribute__(self.noise_distribution,)(
            size=(n_sample, self.n_random_features),
        )

        return pd.concat([x, pd.DataFrame(rnd_ftes, index=x.index)], axis=1)


class BootstrapNegControlSelector(Transformer):
    """Selects features with higher importance than random features over a threshold."""

    def __init__(
        self,
        estimator: Optional[RegressorMixin] = None,
        n_random_features: Optional[int] = 5,
        n_bootstrap_samples: Optional[int] = 25,
        threshold: Optional[float] = 0.1,
        noise_distribution: Optional[str] = "uniform",
        random_state: Optional[int] = 1234,
    ):
        """Instantiate the selector and attach parameters.

        Args:
            estimator: An instantiated estimator to fit to the data and retrieve
                feature importance values from.
            n_random_features: The number of random features to create.
                Default is 5.
            n_bootstrap_samples: The number of bootstrap samples to perform.
                Default is 25.
            threshold: The threshold at which to remove features. If the percentage
                of bootstrap iterations where the most important random feature
                has a higher importance than a certain feature is higher than this
                threshold, this feature will be dropped. The threshold should be
                in [0,1].
            noise_distribution: The distribution to use to sample the random features,
                where it is an attribute of numpy random. Default is uniform.
            random_state: Random state for the random number generator.
        """
        if estimator is None:
            self.estimator = RandomForestRegressor()
        else:
            self.estimator = estimator
        self.n_random_features = n_random_features
        self.noise_distribution = noise_distribution
        self.random_state = random_state
        self.n_bootstrap_samples = n_bootstrap_samples
        self.threshold = threshold

        self.rand_num_gen = np.random.RandomState(self.random_state)

    @property
    def threshold(self):
        """Threshold getter."""
        return self._threshold

    @threshold.setter
    def threshold(self, value_to_set: float):
        """Sets threshold and checks for correct values."""
        if 0 <= value_to_set <= 1:
            self._threshold = value_to_set
        else:
            raise ValueError("The threshold must be between 0 and 1 inclusive.")

    def fit(
        self,
        x: pd.DataFrame,  # noqa: WPS111
        y: Optional[pd.DataFrame] = None,  # noqa: WPS111
        **fit_params,
    ) -> "BootstrapNegControlSelector":
        """Fits the selector.

        Fits the selector by iteratively running an estimator on a bootstrapped version
        of the given dataframe with additional randomly sampled features, computing
        their feature importance and removes those features in the data which
        have a lower importance than the most important random feature in x% of the
        iterations, where x is the input threshold.

        Args:
            x: Dataframe containing the features which we will select from.
            y: The corresponding target values.
            **fit_params: Other estimator specific parameters.

        Returns:
            A fitted selector.

        """
        self.check_x(x)
        feat_imp_counts = self._run_bootstrap(x, y, **fit_params)

        feat_imp_perc = feat_imp_counts / self.n_bootstrap_samples

        mask = np.ones_like(feat_imp_perc, dtype=bool)
        mask[feat_imp_perc > self._threshold] = False

        self.support_ = mask  # noqa: WPS120
        self.selected_features_ = x.columns[self.support_].tolist()  # noqa: WPS120
        return self

    def get_support(self) -> np.ndarray:
        """Provides a boolean mask of the selected features."""
        check_is_fitted(self)
        return self.support_

    def transform(self, x: pd.DataFrame):  # noqa: WPS111
        """Selects the features selected in `fit` from a provided dataframe."""
        check_is_fitted(self)
        self.check_x(x)
        return x.loc[:, self.support_]

    def _run_bootstrap(  # noqa: WPS210
        self,
        x: pd.DataFrame,  # noqa: WPS111
        y: pd.DataFrame,  # noqa: WPS111
        **fit_params,
    ):
        """Run the bootstrap iterations to sample data and fit the estimator."""
        feat_imp_counts = np.zeros(x.shape[1])

        for _ in range(self.n_bootstrap_samples):
            x_bootstrap, y_bootstrap = self._create_bootstrap_data(x, y)

            model = self.estimator
            model.fit(x_bootstrap, y_bootstrap, **fit_params)
            feat_importances = _get_feature_importances(model)

            random_feat_importances = feat_importances[
                -self.n_random_features :  # noqa: E203
            ]
            feat_importances_fte = feat_importances[: -self.n_random_features]

            max_rnd_imp = random_feat_importances.max()

            mask = np.zeros_like(feat_importances_fte)
            mask[feat_importances_fte < max_rnd_imp] = 1

            feat_imp_counts += mask

        return feat_imp_counts

    def _create_bootstrap_data(
        self,
        x: pd.DataFrame,  # noqa: WPS111
        y: pd.DataFrame,  # noqa: WPS111
    ):
        """Sample from the given dataframe and sample random features."""
        n_sample = len(x)

        # resample features
        sampled_indices = np.random.choice(
            np.arange(x.shape[0]),
            size=n_sample,
            replace=True,
        )
        x_bootstrap = x.iloc[sampled_indices]
        y_bootstrap = y.iloc[sampled_indices]

        # sample random features
        rnd_ftes = self.rand_num_gen.__getattribute__(  # noqa: WPS609
            self.noise_distribution,
        )(
            size=(n_sample, self.n_random_features),
        )

        x_bootstrap = pd.concat(
            [
                x_bootstrap,
                pd.DataFrame(rnd_ftes, index=x_bootstrap.index),
            ],
            axis=1,
        )

        return x_bootstrap, y_bootstrap


class NonParametricNegControlSelector(Transformer):
    """Selects features based on a one-sided Wilcoxon test for feature importance."""

    def __init__(
        self,
        estimator: Optional[Estimator] = None,
        significance: Optional[float] = 0.05,
        n_bootstrap_samples: Optional[int] = 25,
        noise_distribution: Optional[str] = "uniform",
        random_state: Optional[int] = 1234,
    ):
        """Instantiate the selector and attach parameters.

        Args:
            estimator: An instantiated estimator to fit to the data and retrieve
                feature importance values from.
            significance: The significance/p-value at which to reject the null
                hypothesis. Default is 0.05, giving a 5% significance.
            n_bootstrap_samples: The number of bootstrap samples to perform.
                Default is 25.
            noise_distribution: The distribution to use to sample the random
                features, where it is an attribute of numpy random. Default is uniform.
            random_state: Random state for the random number generator.

        """

        self.estimator = RandomForestRegressor() if estimator is None else estimator
        self.noise_distribution = noise_distribution
        self.random_state = random_state
        self.n_bootstrap_samples = n_bootstrap_samples
        self.significance = significance
        self.x_cols = None
        self.rand_num_gen = np.random.RandomState(self.random_state)
        self.random_feat_name = "random_fte"

    @property
    def significance(self):
        """Significance getter."""
        return self._significance

    @significance.setter
    def significance(self, value_to_set: float):
        """Sets significance value and checks for correct values."""
        if 0 <= value_to_set <= 1:
            self._significance = value_to_set
        else:
            raise ValueError(
                "The significance or p-value must be between 0 and 1 inclusive.",
            )

    def fit(
        self,
        x: pd.DataFrame,  # noqa: WPS111
        y: Optional[pd.DataFrame] = None,  # noqa: WPS111
        **fit_params,
    ) -> "NonParametricNegControlSelector":
        """Fits the selector.

        Runs bootstrapped samples of the input dataframe and a random feature (negative
        control), fits the estimator and collects the feature importances across
        iterations. Then, for each feature and negative control pair, performs a
        one-sided Wilcoxon signed rank test to test whether the feature is ranked
        higher than the control vs a null hypothesis of the same or lower rank.

        Args:
            x: Dataframe containing the features which we will select from.
            y: The corresponding target values.
            **fit_params: Other estimator specific parameters.

        Returns:
            A fitted selector.

        """
        self.check_x(x)
        self.x_cols = x.columns.astype(str).tolist()

        bootstrap_importances = self._run_bootstrap(x, y, **fit_params)
        feat_significances = self._sign_test(bootstrap_importances)

        mask = np.ones_like(feat_significances, dtype=bool)
        mask[feat_significances > self.significance] = False

        self.support_ = mask  # noqa: WPS120
        self.selected_features_ = x.columns[self.support_].tolist()  # noqa: WPS120
        return self

    def get_support(self) -> np.ndarray:
        """Provides a boolean mask of the selected features."""
        check_is_fitted(self)
        return self.support_

    def transform(self, x: pd.DataFrame):  # noqa: WPS111
        """Selects the features selected in `fit` from a provided dataframe."""
        check_is_fitted(self)
        self.check_x(x)
        return x.loc[:, self.support_]

    def _sign_test(self, bootstrap_importances: pd.DataFrame):
        """Performs the one-sided Wilcoxon signed rank test."""
        num_ftes = len(self.x_cols)
        feat_significances = np.zeros(num_ftes)
        for feature_idx in range(num_ftes):
            col = self.x_cols[feature_idx]
            _, p_value = wilcoxon(
                bootstrap_importances[col],
                bootstrap_importances[self.random_feat_name],
                alternative="greater",
            )
            feat_significances[feature_idx] = p_value
        return feat_significances

    def _run_bootstrap(
        self,
        x: pd.DataFrame,  # noqa: WPS111
        y: pd.DataFrame,  # noqa: WPS111
        **fit_params,
    ):
        """Run the bootstrap iterations to sample data and fit the estimator."""
        bootstrap_importances = pd.DataFrame(
            {},
            columns=self.x_cols + [self.random_feat_name],
            index=range(self.n_bootstrap_samples),
        )

        for sample_idx in range(self.n_bootstrap_samples):
            x_bootstrap, y_bootstrap = self._create_bootstrap_data(x, y)

            model = self.estimator
            model.fit(x_bootstrap, y_bootstrap, **fit_params)

            bootstrap_importances.loc[sample_idx, :] = _get_feature_importances(model)

        return bootstrap_importances

    def _create_bootstrap_data(
        self,
        x: pd.DataFrame,  # noqa: WPS111
        y: pd.DataFrame,  # noqa: WPS111
    ):
        """Sample from the given dataframe and sample a random feature."""
        n_sample = len(x)

        # resample features
        sampled_indices = np.random.choice(
            np.arange(x.shape[0]),
            size=n_sample,
            replace=True,
        )
        x_bootstrap = x.iloc[sampled_indices].copy()
        y_bootstrap = y.iloc[sampled_indices].copy()

        # sample random feature
        rnd_fte = self.rand_num_gen.__getattribute__(  # noqa: WPS609
            self.noise_distribution,
        )(size=n_sample)

        x_bootstrap[self.random_feat_name] = rnd_fte

        return x_bootstrap, y_bootstrap


def _get_feature_importances(estimator):
    """Retrieve feature importances from estimator."""
    importances = getattr(estimator, "feature_importances_", None)
    coef_ = getattr(estimator, "coef_", None)  # noqa: WPS120

    if importances is None and coef_ is not None:
        importances = np.abs(coef_)

    elif importances is None:
        class_name = estimator.__class__.__name__
        raise AttributeError(
            f"The underlying estimator {class_name}"
            " has no `coef_` or `feature_importances_` attribute."
            " Either pass a fitted estimator or call fit before calling transform.",
        )

    return importances
