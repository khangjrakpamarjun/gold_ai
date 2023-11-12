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
Implementation of MonteCarloSolver solver

"""

import logging
from numbers import Real
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from optimizer.exceptions import SolutionNotFoundError
from optimizer.solvers.base import Solver
from optimizer.types import Matrix, Vector

_OPTIMISATION_MIN_STR = "Optimisation min"
_OPTIMISATION_MAX_STR = "Optimisation max"
logger = logging.getLogger(__name__)


class MonteCarloSolver(Solver):
    """
    Implementation of MC optimizer for a single subarea.
    This algorithm uses historical distribution to generate random
    samples and the best one is selected at the end. This allows the algorithm to
    implicitly model underlying dependencies between the controls (i.e. if water
    level goes up, usually feed increases as well)

    This algorithm is less generic than other ones in this package and mainly
    written for Optimus AI use case. However, it can be adapted to other use cases.

    This class uses scipy `multivariate_normal` sampler, however other libraries
    have been successfully used in the past, such as `emcee.EnsembleSampler`.
    Sampling can be changed by overriding `_generate_samples` method, if
    another sampler produces better results.

    This algorithm is best fitted to handle low number of controls, typically
    less than 10, due to the curse of dimensionality. If more controls present,
    splitting it into independent sub-problems typically produces great results.

    Algorithm works as follows (Optimus AI specific steps):
        1. Before algorithm is run, historical samples are calculated using HSS
            (Historical Similarity Sphere), see `NeighbourhoodCalculator` class
        2. Historical samples are taken as a prior, so that sampler can draw
            from the same distribution. Mean and covariance are calculated
        3. Samples are generated using sampling algorithm (scipy in this case)
        4. Samples are de-duped and filtered by boundary conditions. If this step
            filters too many samples, try changing sampling algorithm to MCMC
            or relax boundary conditions
        5. Samples are evaluated (outside of the ask/tell framework of this class)
        6. Samples are repaired (outside of the ask/tell framework of this class)
        7. Samples are filtered based on hard constraints. (outside ask-tell loop).
        8. Additional filtering is beneficial but based on the use case, that
            removed objective value components (such as throughput and recovery)
            that are extreme. This is usually done by filtering in P70-P95 range
            (outside of the ask/tell framework of this class)
        9. Best solution is selected based on best objective function

    Example:
        ::

            >>> hss_calc = NeighbourhoodCalculator(train_set, granular_df,
            ...                                    distance_dims=input_dims)
            >>> sphere_df = hss_calc.get_points(current_plant_state,
            ...        n_historical=30)

            >>> problem = MillOptimisationProblem( # Defined separately, by the user
            ...     model=model,
            ...     model_features=model_features,
            ...     current_plant_state=current_plant_state,
            ...     optimizable_columns=control_features
            ... )

            >>> mc_solver = MonteCarloSolver(
            ...     control_features=control_features,
            ...     sphere_df=sphere_df,
            ...     n_samples=1000,
            ...     domain=bounds,
            ...     seed=42
            ... )

            >>> while not mc_solver.stop():
            >>>     solutions = mc_solver.ask()
            >>>     objectives, solutions = problem(solutions)
            >>>     mc_solver.tell(solutions, objectives)

            >>> solution, objective = mc_solver.best()
            >>> solution_dataframe = mc_solver.best_as_dataframe()

    """

    def __init__(
        self,
        control_features: List[str],
        sphere_df: pd.DataFrame,
        n_samples: int,
        domain: List[Tuple[Real, Real]],
        sense: str = "minimize",
        seed: int = None,
    ):
        """Constructor

        Args:
            control_features (List[str]): control features
            sphere_df (pd.DataFrame): neighbourhood sphere that is used
                to create distribution
            n_samples (int): number of samples to create
            domain (List[Tuple]): control variable bounds
            sense (str, optional): minimization or maximization problem.
                Defaults to "minimize".
            seed (int, optional): random seed. Defaults to None.

        Raises:
            SolutionNotFoundError: When no viable solution found from samples.

        """
        self._controls_df = self._drop_constant_column(sphere_df[control_features])
        if self._controls_df.shape[1] != sphere_df[control_features].shape[1]:
            raise ValueError(
                "The sphere_df contains control variables with constant "
                "values. These will have zero covariance with other "
                "features, and cannot be used to generating samples."
            )
        self._control_features = self._controls_df.columns.tolist()
        super().__init__(domain, sense=sense, seed=seed)

        self._mean = self._controls_df.mean().values
        self._covariance_matrix = self._controls_df.cov()
        self._ndimensions = self._controls_df.shape[1]
        self._n_samples = n_samples
        self._final_solutions = np.array([])
        self._final_objectives = np.array([])
        self._is_sampled = False  # True when ask is called

        self._init_bounds(domain, control_features)

    def _init_bounds(self, bounds: List[Tuple], control_features: List[str]):
        """
        Create bounds dataframe for usage within _filter_within_bounds

        Args:
            bounds: List of bounds for each dimension
            control_features: List of names for each dimension


        """
        named_bounds = list(zip(*bounds))
        named_bounds.append(control_features)
        self._bounds = pd.DataFrame(
            list(zip(*named_bounds)),
            columns=[_OPTIMISATION_MIN_STR, _OPTIMISATION_MAX_STR, "tag"],
        ).set_index("tag")

    def _drop_constant_column(self, df: pd.DataFrame):
        """Drops constant value columns of pandas dataframe.

        Args:
            df (pd.DataFrame): data frame to drop columns

        Returns:
            [pd.DataFrame]: dataframe with constant columns dropped
        """
        return df.loc[:, (df != df.iloc[0]).any()]

    def _filter_within_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies filter bounds on the `df`

        Arguments:
            df {pd.DataFrame} -- initial data frame

        Returns:
            pd.DataFrame -- filtered data frame
        """
        for col in list(self._bounds.index):
            df = df.loc[
                (df[col] >= self._bounds.loc[col][_OPTIMISATION_MIN_STR])
                & (df[col] <= self._bounds.loc[col][_OPTIMISATION_MAX_STR])
            ]

        return df

    def ask(self) -> np.ndarray:
        """Ask step of the algorithm

        Returns: current parameter values.
        """
        mc_sample = self._generate_samples()

        mc_sample.drop_duplicates(inplace=True)
        logger.debug(
            f"After removing duplicates there are {mc_sample.shape[0]} samples"
        )

        mc_sample_bounds_filtered = self._filter_within_bounds(mc_sample)
        logger.debug(
            f"After bounds filter there are {mc_sample_bounds_filtered.shape[0]} "
            f"samples"
        )

        return mc_sample_bounds_filtered.values

    def stop(self) -> bool:
        """Determine if the solver should terminate or not.
        For this solver, it always stops after one iteration

        Returns:
            Bool, True is the solver should terminate.
        """
        return self._is_sampled

    def best(self) -> Tuple[np.ndarray, float]:
        """Get the current best solution.

        Returns:
            Tuple[np.ndarray, float]: the best solution and its objective value.
        """
        if self._is_sampled and self._final_solutions.shape[0] > 0:
            idx_min = np.argmin(self._final_objectives)
            return self._final_solutions[idx_min], self._final_objectives[idx_min]
        else:
            logger.warning("No solutions found")
            raise SolutionNotFoundError(
                "No solutions found. All solutions generated"
                "from historical data fell outside of the "
                "feasible region. Increasing n_samples, or "
                "decreasing the dimensionality of your problem"
                "can help."
            )

    def tell(self, parameters: Matrix, objective_values: Vector):
        """Tell step of the solver

        Args:
            parameters: Matrix of updated solution values to pass to the solver.
            objective_values: Vector of updated objective values to pass to the solver.
        """
        parameters, objective_values = self._process_told_parameters_and_objectives(
            parameters, objective_values
        )
        self._final_solutions = parameters
        self._final_objectives = objective_values
        self._is_sampled = True

    @property
    def parameters(self):
        return self._final_solutions

    @property
    def _internal_objective_values(self) -> Vector:
        return self._final_objectives

    def _generate_samples(self) -> pd.DataFrame:
        """Generates MVN samples based on mean and covariance matrix

        Returns:
            pd.DataFrame: data frame with samples
        """
        samples = multivariate_normal.rvs(
            self._mean,
            self._covariance_matrix,
            size=self._n_samples,
            random_state=self.seed,
        )
        mc_sample = pd.DataFrame(samples)
        logger.debug(f"{mc_sample.shape[0]} samples created")
        mc_sample.columns = self._control_features
        return mc_sample
