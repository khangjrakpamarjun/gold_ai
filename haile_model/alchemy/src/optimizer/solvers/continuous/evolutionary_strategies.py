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
Covariance Matrix Adaptation-Evolutionary Strategies solver.

Also see a tutorial on this algorithm here:
    https://arxiv.org/pdf/1604.00772.pdf

There's a lot going on there. Skip to Appendix A for a summary and then
    see Appendix C for a MATLAB implementation.
"""

from copy import deepcopy
from numbers import Real
from typing import List, Tuple, Union

import numpy as np

from optimizer.domain import RealDimension
from optimizer.exceptions import InitializationError, MaxIterationError
from optimizer.solvers.continuous.base import ScaledContinuousSolver
from optimizer.types import Matrix, Vector
from optimizer.utils.initializer import uniform_random


class EvolutionaryStrategiesSolver(
    ScaledContinuousSolver
):  # pylint: disable=too-many-instance-attributes
    """
    (mu/mu_W, lambda)-CMA-Evolutionary Strategies.

    Most of the variables in this code will follow their definitions found here:
        https://arxiv.org/pdf/1604.00772.pdf
    """

    def __init__(
        self,
        domain: List[Union[Tuple[Real, Real], RealDimension]],
        sense: str = "minimize",
        seed: int = None,
        initial_mean: Vector = None,
        mu: int = None,
        lam: int = None,
        maxiter: int = 1000,
        sigma: float = 0.3,
    ):
        """Constructor.

        This constructor has a _lot_ of notation and setup.
        Skip down to the ask/tell method for the meat of the algorithm.

        Args:
            domain: list of tuples, upper and lower domain for each dimension.
            sense: string, optional sense string for "minimize" or "maximize".
            seed: int, optional random seed.
            initial_mean: optional Vector for the initial mean.
                Will be randomly chosen if not specified.
            mu: number of points selected from the generated sample.
                If not specified, this will be lam // 2.
            lam: number of points sampled each generation.
                If not specified, this will be 4 + floor(3 * ln(d)) for dimension d.
            maxiter: maximum number of iterations.
            sigma: initial standard deviation. Essentially a step size for sampling.

        Raises:
            ValueError: if mu greater than lam.
            ValueError: if the initial mean is outside the domain of the problem.
        """
        super(EvolutionaryStrategiesSolver, self).__init__(
            domain=domain, sense=sense, seed=seed
        )

        n = len(domain)

        # Set the number of offspring.
        if lam is None:
            lam = int(4 + np.floor(3 * np.log(n)))

        # Set the number of parents chosen.
        if mu is None:
            mu = lam // 2

        if mu > lam:
            raise ValueError(
                "Provided mu must be less than or equal to lam. "
                "In other words, the number of chosen points must "
                "be less or equal to the number of points sampled each generation."
            )

        self.mu = mu
        self.lam = lam

        # Set the recombination weights.
        # This strategy disregards the negative weights described in Hansen 2016 and
        # only considers applying weights to the best mu samples.
        weights = np.log(self.mu + 0.5) - np.log(np.arange(self.mu) + 1)
        self.weights = weights / np.sum(weights)

        # "Variance-effectiveness" factor.
        self.mu_eff = 1 / np.dot(self.weights, self.weights)

        # Set cumulation and learning rates for the covariance and step size.
        self.cov_cumulation = (4.0 + self.mu_eff / n) / (
            n + 4.0 + 2.0 * self.mu_eff / n
        )

        self.sig_cumulation = (self.mu_eff + 2.0) / (n + self.mu_eff + 5.0)

        self.rank_1_lr = 2.0 / ((n + 1.3) * (n + 1.3) + self.mu_eff)

        self.rank_mu_lr = (
            2.0
            * (self.mu_eff - 2.0 + 1.0 / self.mu_eff)
            / ((n + 2.0) * (n + 2.0) + self.mu_eff)
        )

        # Set damping coefficient for step size.
        self.damp_sig = (
            1.0
            + 2.0 * max(0, np.sqrt(self.mu_eff - 1.0) / (n + 1.0) - 1.0)
            + self.sig_cumulation
        )

        # Precompute the expected value of the norm of a random vector from N(0, I).
        self.expected_norm = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        # Step size.
        self.sigma = sigma

        # Mean of multivariate normal.
        # If not supplied, randomly sample from [0, 1]^dimension.
        # First, if provided, check that the inital mean is within with given domain.
        if initial_mean is not None:
            initial_mean = self._transform_parameters(initial_mean)

            if not all((initial_mean >= 0) & (initial_mean <= 1)):
                raise InitializationError(
                    "Provided initial mean must be within the domain of the problem."
                )

            self._mvn_mean = initial_mean

        else:
            self._mvn_mean = uniform_random((n,), self.rng)

        # Covariance of multivariate normal.
        # Defined as the left eigendecomposition of the actual covariance.
        # In other words our covariance, C = BD^2B, we'll just keep track of B and D
        # and then scale the samples we get from N(0, I) using these matrices.
        self.eigenvectors = np.eye(n)  # B matrix.
        self.eigenvalues = np.eye(n)  # D matrix.
        self.covariance = np.eye(n)
        self.invsqrtC = np.eye(n)  # Inverse square root of covariance.

        # Evolution paths.
        self.path_sigma = uniform_random((n,), self.rng)
        self.path_cov = uniform_random((n,), self.rng)

        # Book keeping parameters.
        self.maxiter = maxiter
        self.niters = 1
        self.nevals = 0
        self.eigen_update = 0
        self.best_sample = None
        self.best_objective = None
        self.elites = None
        self.elites_objectives = None
        self.current_samples = None

        self.dim = n

        # pylint: enable=too-many-instance-attributes

    def ask(self) -> np.ndarray:
        """Get the parameters of the current population sample.

        Returns:
            np.array of sample of values to evaluate.

        Raises:
            MaxIterationError: when called after maximum iterations.
            InitializationError: when called without evaluating the first sample.
                Not strictly necessary, but is a good check to ensure the
                solver loop is evaluating solutions properly.
        """
        self.niters += 1

        if self.niters > self.maxiter:
            raise MaxIterationError(
                f"Evolutionary strategies cannot exceed its "
                f"max iteration ({self.maxiter})."
            )

        if self.told and self.elites_objectives is None:
            raise InitializationError(
                "Attempted to generate a new population without evaluating first."
            )

        # Check if ask has been called twice without giving objective values.
        if not self.told and self.current_samples is not None:
            return self.current_samples

        samples = self.rng.multivariate_normal(
            np.zeros(self.dim), np.eye(self.dim), size=(self.lam,)
        )

        samples = self._clip(
            self._mvn_mean
            + self.sigma * (self.eigenvectors @ self.eigenvalues @ samples.T).T
        )

        self.told = False

        self.current_samples = self._inverse_transform_parameters(samples)

        return self.current_samples

    def tell(self, parameters: Matrix, objective_values: Vector):
        """Set the population and objective values.
        Updates internal distribution parameters.

        Args:
            parameters: Matrix of parameter values representing the population.
            objective_values: Vector of objective values.
        """
        super(EvolutionaryStrategiesSolver, self).tell(parameters, objective_values)

        self.nevals += self.lam

        parameters, objective_values = self._process_told_parameters_and_objectives(
            parameters, objective_values
        )

        # Put parameters back into [0, 1] scale
        # and remove the current mean and standard deviation.
        parameters = self._transform_parameters(parameters)

        self.update_elites_and_bests(parameters, objective_values)

        # Compute new mean.
        old_mean = deepcopy(self._mvn_mean)
        self._mvn_mean = sum(self.elites[i] * self.weights[i] for i in range(self.mu))

        # Update evolution paths.
        self.path_sigma = (1 - self.sig_cumulation) * self.path_sigma + np.sqrt(
            self.cov_cumulation * (2 - self.cov_cumulation) * self.mu_eff
        ) * self.invsqrtC @ (self._mvn_mean - old_mean) / self.sigma

        hsig = (
            np.linalg.norm(self.path_sigma) ** 2
            / self.dim
            / (1 - (1 - self.sig_cumulation) ** (2 * self.nevals / self.lam))
        ) < (2 + 4 / (self.dim + 1))

        self.path_cov = (1 - self.cov_cumulation) * self.path_cov + hsig * np.sqrt(
            self.cov_cumulation * (2 - self.cov_cumulation) * self.mu_eff
        ) * (self._mvn_mean - old_mean) / self.sigma

        # Update covariance matrix.
        old_cov = deepcopy(self.covariance)

        self.covariance = sum(
            np.outer(self.elites[i] - old_mean, self.elites[i] - old_mean)
            * self.weights[i]
            for i in range(self.mu)
        )

        self.covariance /= self.sigma * self.sigma

        self.covariance = (
            (1 - self.rank_1_lr - self.rank_mu_lr) * old_cov
            + self.rank_mu_lr * self.covariance
            + self.rank_1_lr
            * (
                np.outer(self.path_cov, self.path_cov)
                + (1 - hsig) * self.cov_cumulation * (2 - self.cov_cumulation) * old_cov
            )
        )

        # Update standard deviation.
        self.sigma *= np.exp(
            min(
                0.6,
                (self.sig_cumulation / self.damp_sig)
                * (np.linalg.norm(self.path_sigma) / self.expected_norm - 1),
            )
        )

        # Perform the eigendecomposition of the covariance if necessary.
        if (
            self.nevals - self.eigen_update
            > self.lam / (self.rank_1_lr + self.rank_mu_lr) / self.dim / 10.0
        ):
            self.update_eigen_decomposition()

        self.told = True

    def update_elites_and_bests(self, parameters: Matrix, objective_values: Vector):
        """Update the current and historical best population.

        Here "elite" refers to the mu best solutions from the lambda many
        samples generated in the ask() sampling step.

        Args:
            parameters: Matrix of parameter values representing the population.
            objective_values: Vector of objective values.
        """
        # Get the best mu samples from the population.
        best_idxs = np.argsort(objective_values)[: self.mu]
        self.elites = parameters[best_idxs]
        self.elites_objectives = objective_values[best_idxs]

        # Update the current best if necessary.
        if (
            self.best_objective is None
            or self.elites_objectives[0] < self.best_objective
        ):
            self.best_objective = self.elites_objectives[0]
            self.best_sample = self.elites[0]

    def update_eigen_decomposition(self):
        """Update the internal eigenvalues, eigenvectors, and covariance inverse."""
        self.eigen_update = self.nevals

        try:
            # Ensure covariance is symmetric.
            sym = np.triu(self.covariance) + np.triu(self.covariance, 1).T

            values, vectors = np.linalg.eig(sym)

            # Any complex parts returned are the result of floating point errors.
            # Just strip them off if they do occur.
            values, vectors = np.real(values), np.real(vectors)
            values = np.maximum(1e-20, values)
            values_inv = 1 / deepcopy(values)
            values = np.diag(values)

            self.eigenvalues = values
            self.eigenvectors = vectors
            self.invsqrtC = (
                self.eigenvectors @ np.diag(values_inv) @ self.eigenvectors.T
            )

        except np.linalg.LinAlgError:
            pass  # Keep old values if our estimate does not converge.

    def stop(self) -> bool:
        """Determine if we should stop iterating the Ask and Tell loop.

        Returns:
            Boolean, True if we should stop.
        """
        return self.niters >= self.maxiter

    def best(self) -> Tuple[np.ndarray, float]:
        """Get the best solution and its objective value.

        Returns:
            Vector and float, the solution vector and its objective value.
        """
        return (
            self._inverse_transform_parameters(self.best_sample),
            self.best_objective
            if self.sense == "minimize"
            else -1 * self.best_objective,
        )

    @property
    def _internal_objective_values(self) -> Vector:
        """Get internal objective values.

        Returns:
            Vector of the current objective values.
        """
        return self.elites_objectives

    @property
    def parameters(self) -> np.ndarray:
        """Get the current parameter values.

        Returns:
            Matrix of current parameters.
        """
        return self._inverse_transform_parameters(self.elites)

    @property
    def mean(self) -> np.ndarray:
        """Get the mean of the current multivariate normal distribution.

        Returns:
            Vector.
        """
        return self._inverse_transform_parameters(self._mvn_mean)
