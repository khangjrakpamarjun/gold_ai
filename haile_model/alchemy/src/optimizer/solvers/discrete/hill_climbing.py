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
Discrete local search Solver
"""

from typing import Hashable, List, Tuple, Union

import numpy as np
from sklearn.utils import check_random_state

from optimizer.domain import CategoricalDimension, IntegerDimension
from optimizer.exceptions import MaxIterationError
from optimizer.solvers.discrete.base import IndexedDiscreteSolver
from optimizer.types import Matrix, Vector

# pylint: disable=W0105
"""
    Implementation details (for interested users)

    The algorithm performs local search on each of the solutions in self._s, these are
    referred to as 'current' or 'local' solutions throughout the code. The objective
    values for these solutions are stored in self._obj_values.

    In a single ask & tell cycle we evaluate one single neighbor for each solution. If
    the objective value for a neighbor is better, then we update self._s and
    self._obj_values with this neighbor and it's objective value. If it does not improve
    the solution, we continue exploring the neighborhood of that solution in further
    iterations.

    In this implementation a neighbor of a solution is defined by moving a single
    variable one step. For example, if we have variables a, b and c; which can take
    values a_1, ..., a_na; b_1, ..., b_nb; and c_1, ..., c_nc respectively, then a
    neighbor of a solution (a_1, b_3, c_5) might be (a_2, b_3, c_5) or (a_1, b_3, c_4).
    Therefore, to explore the entire neighborhood of a single solution takes at most
    O(n_variables) ask & tell iterations.

    We do not look for the best neighbor possible to update a solution in self._s; it's
    enough to improve the objective value.

    As soon as we improve a solution or find a local optima, we check for the best
    solutions we've seen so far and update the solutions and values as needed. The best
    seen solutions are stored in self._best_s and self._best_obj_values. These are then
    used for returning the optimal value to the user via self.best().

    The implementation is fully vectorized, and to achieve this we keep several index
    matrices to make this efficiently. In what follows we describe briefly how this is
    implemented.

    The first observation is that self._s and self._best_s keep the solutions in an
    indexed format. This means that entry s_(i, j) is an int, and tells us the position
    in the domain of the j-th variable for the i-th solution. For example, if the domain
    of variable x_3 is [100, 150, 200] and _s(1,3) = 1; this means that the solution 1
    has x_3=150 (we're indexing from 0).

    For each solution we might be exploring a different variable at the same point in
    time. To keep track of what are we exploring we use the matrix self._cm (cm stands
    for current moves). This is a {-1, 0, 1}-matrix, that we add to self._s to get the
    current explored neighbor. Note that self._cm has only one 1 or -1 per row because
    of our definition of neighborhood. Thus, in ask() we simply return self._s +
    self._cm, transforming the index representation into a user-space domain.

    We want to control the biases in which we explore the search space. We avoid
    exploring variables in the same order for all solutions by having an indirection
    index matrix in self._orders. Each row in this matrix corresponds to a solution slot
    in self._s. If the row in self._orders is [3, 0, 2, 1], then we first explore moves
    in the 3rd parameter column (x_3), then x_0, then x_2 and finally x_1. We keep track
    of what column in self._orders we explore for each solution with the vector
    self._cvars (cvars stands for current vars). Thus, for the given order as example,
    we begin the local search by increasing or decreasing the x_3 value index in
    self._cm. After that is completed, we explore increasing or decreasing x_0, and so
    forth. When we finish with the exploration of x_1, and we were not able to improve
    the corresponding solution in self._s, then it means that the solution is locally
    optimal. We now reset the solution to a random one, set the counter in self._cvars
    to 0, and start again with x_3 for this fresh solution. Note that other entries in
    self._orders will yield different orders.
"""


class HillClimbingSolver(IndexedDiscreteSolver):
    """
    Implements a local search algorithm for a given search space domain.
    This is a more direct way of exploring discrete solutions than using a continuous
    algorithm with repair functions. One iteration of this solver is typically faster
    than one iteration of ParticleSwarm with repairs. However, it requires many more
    iterations to find comparable solutions.

    Since the solutions found by this algorithm are local optimas, under a coarse mesh
    this algorithm is likely to find better solutions than continuous algorithms with
    repairs. If the mesh or discrete domain is very fine-grained, then this algorithm
    is not the best choice, since de-facto the solution space is continuous-like.

    This solver does not support a modification of parameters in the self.tell function.
    If parameters are modified with a repair function, this will raise an exception.
    """

    def __init__(
        self,
        domain: List[Union[List[Hashable], IntegerDimension, CategoricalDimension]],
        sense: str = "minimize",
        seed: int = None,
        popsize: int = 1000,
        maxiter: int = 1000,
    ):
        """Constructor

        Args:
            domain: a list of lists that tells which values can each dimension take.
            sense: string, optional sense string for "minimize" or "maximize".
            seed: int, optional random seed.
            popsize: int, number of solutions that are locally improving
            maxiter: int, number of total iterations

        Raises:
            ValueError: if domain are not all numeric types.
            ValueError: if domain is not a list of tuples.
        """
        # For supporting the abstract interface we infer domain from domain
        super().__init__(domain, sense, seed)

        self._iterations_left = maxiter

        self._pop = popsize
        self._seed = seed
        self._rng = check_random_state(seed)

        dim = self._pop, self.domain_matrix.shape[0]

        # The indices of the current solutions. Dimensions are pop x nvariables.
        # The value of variable j for solution i is domain[j][s[i, j]]
        self._s = np.zeros(dim, dtype=int)

        # Objective values for current s (without any moves, just as s is stored)
        self._obj_values = np.inf + np.zeros(self._pop, dtype=int)

        # keeps track of the best seen solutions, in the same format as self._s
        self._best_s = np.zeros(dim, dtype=int)

        # keeps track of the best objective values seen so far
        self._best_obj_values = np.inf + np.zeros(self._pop, dtype=int)

        # randomly initialize solutions
        self._reset_solutions(np.array([True] * self._pop))

        # current vars for each solution. This tells us which variable are we trying
        # to move for improving locally the solution. cvars[i] = j means that we're
        # moving dimension self._orders[i, j] for solution i.
        self._cvars = np.zeros(self._pop, dtype=int)

        # current moves: self._cm[i, j] = 1 means that we tell/told the user a set of
        # parameters that increases by one the index s[i,j].
        # Moves in current implementation are: -1, 0, 1. Most of matrix is 0.
        self._cm = np.zeros(dim, dtype=int)

        # 10000 is to avoid being biased by using argsort to explore lower range of the
        # variable indexes. If we call argsort on 0,0,0,0,0, we'll get 0,1,2,3,4;
        # we thus need to increase the upper random bound so we reduce duplicates
        self._orders = np.argsort(self._rng.randint(10000 * dim[1], size=dim), axis=1)

    def _reset_solutions(self, boolean_index_vector: Vector):
        """Generates random solutions in self._s for positions that are True in the
        boolean_index_vector.

        Args:
            boolean_index_vector: Boolean vector used as a mask for the operation.
        """
        # Disabling pylint unsubscriptable-object error, see
        # https://github.com/PyCQA/pylint/issues/3139
        ncols = self._s.shape[1]  # pylint: disable=E1136
        nrows = boolean_index_vector.sum()
        self._s[boolean_index_vector] = self._rng.randint(
            0, self.domain_lengths, size=(nrows, ncols)
        )
        self._obj_values[boolean_index_vector] = np.inf

    def _reset_current_moves(self, boolean_index_vector: Vector):
        """Clears the current moves matrix self._cm in positions where the
        boolean_index_vector is True.

        Args:
            boolean_index_vector: Boolean vector used as a mask for the operation.
        """
        self._cm[boolean_index_vector] = 0

    def _reset_exploration_indices(self, boolean_index_vector: Vector):
        """Sets to 0 the self._cvars on the positions where the boolean_index_vector is
        True. This effectively resets the variable exploration for each solution.

        Args:
            boolean_index_vector: Boolean vector used as a mask for the operation.
        """
        self._cvars[boolean_index_vector] = 0

    def _inc_dec_zero_move(self, current_neighbor: np.ndarray) -> np.ndarray:
        """Given a current neighbor move vector (this is a {-1, 0, 1}-vector), it
        transitions each dimension from one exploration direction to the next one.
        Currently, we go from 0 -> 1 -> -1 -> 0. It is responsibility of the user of
        this function to check the boundaries for each dimensions, this is just a helper

        Args:
            current_neighbor: an {-1, 0, 1}-ndarray

        Returns:
            Same dimension array with one transition step applied to all of its values.

        """
        completed = current_neighbor == -1
        current_neighbor = np.where(current_neighbor == 1, -1, current_neighbor)
        current_neighbor = np.where(current_neighbor == 0, 1, current_neighbor)
        current_neighbor[completed] = 0
        return current_neighbor

    def _transition_current_move(self):
        """Updates self._cm (current moves matrix) based on the the currently explored
        variables (by using self._cvars and self._order).
        It also updates the counters in self._cvars for the variables that have been
        locally explored (with moves (+-1)).
        If no more moves are available for a solution, it will set the current move row
        associated to that solution to 0. It is responsibility of the user to treat the
        rows in 0, since the next call would generate all the moves again.
        """
        # Disabling pylint unsubscriptable-object error, see
        # https://github.com/PyCQA/pylint/issues/3139
        nsols = self._cm.shape[0]  # pylint: disable=E1136
        z2n = np.arange(nsols)  # zero to n vector, helper for indexing
        cur_dim_to_explore = self._orders[z2n, self._cvars]
        cur_domain_indices = self._s[z2n, cur_dim_to_explore]
        current_dom_upper_bounds = self.domain_lengths[cur_dim_to_explore]

        # Transition between 0, 1 and -1 to 1, -1, 0, respectively

        # Make sure the transitions are within domain: we transition at most 3 times
        # each position. If after 3 transitions we're still out of domain there's a
        # state error and we'll raise an exception
        tr = self._inc_dec_zero_move(self._cm[z2n, cur_dim_to_explore])
        within_domain = (tr + cur_domain_indices >= 0) & (
            tr + cur_domain_indices < current_dom_upper_bounds
        )

        tr[~within_domain] = self._inc_dec_zero_move(tr[~within_domain])
        within_domain = (tr + cur_domain_indices >= 0) & (
            tr + cur_domain_indices < current_dom_upper_bounds
        )

        tr[~within_domain] = self._inc_dec_zero_move(tr[~within_domain])
        within_domain = (tr + cur_domain_indices >= 0) & (
            tr + cur_domain_indices < current_dom_upper_bounds
        )

        if not within_domain.all():
            raise RuntimeError(
                "Transitioning in local search error. Were not able to"
                "transition into a valid state. Some dimension is out of the domain."
            )

        self._cm[z2n, cur_dim_to_explore] = tr

        # If transition = 0, then the current dimension tried all it's moves.
        # Increase search variable for that solution
        self._cvars[tr == 0] += 1

        # Reset the search for the solution that exhausted all dimensions.
        exhausted_variables = self._cvars >= len(self.domain_lengths)
        self._reset_solutions(exhausted_variables)
        self._reset_exploration_indices(exhausted_variables)

        # If all went well, the current moves matrix on exhausted searches must be all 0
        if not (self._cm[self._cvars >= len(self.domain_lengths)] == 0).all():
            raise RuntimeError("Arrived to an invalid state.")

    def _update_current_solutions_and_objective(self, objective_values: np.array):
        """Based on the objective values for self._s and the ones just explored (given
        by self._cm+self_s and the parameter objective_values), updates the improved
        solutions in self._s and the improved objective values in self._obj_values.

        Args:
            objective_values: numpy array with objective values for self._cm + self_s.

        """
        improved_solutions = objective_values < self._obj_values
        self._obj_values[improved_solutions] = objective_values[improved_solutions]
        self._s[improved_solutions] = (self._s + self._cm)[improved_solutions]

    def _update_best_solutions_and_objective(self):
        """Updates best seen solutions and objective values (row-wise) by comparing them
        with the current _obj_values and _s. If the best solution so far has a higher
        objective value than the one stored in self._s, then update. Also update the
        best objective value seen so far.
        """
        improved_solutions = self._obj_values < self._best_obj_values
        self._best_obj_values[improved_solutions] = self._obj_values[improved_solutions]
        self._best_s[improved_solutions] = self._s[improved_solutions]

    def _incorporate_results_for_current_move(self, objective_values: np.ndarray):
        """Given new objective values corresponding to the current exploration (self._s+
        self._cm), incorporate this information by updating current solutions, current
        objective values, best solutions and best objective values.

        It also resets the exploration for each improved solution.

        Args:
            objective_values: numpy array with objective values for self._cm + self_s.
        """
        current_improved_solutions = objective_values < self._obj_values

        self._update_current_solutions_and_objective(objective_values)
        self._update_best_solutions_and_objective()

        # Reset the search in improved (current) solutions
        self._reset_current_moves(current_improved_solutions)
        self._reset_exploration_indices(current_improved_solutions)

    def ask(self) -> np.ndarray:
        """Get the current parameters of population.

        Returns:
            np.array of current population values.

        Raises:
            MaxIterationError: when called after maximum iterations.
        """
        if self.stop():
            raise MaxIterationError("Exceeded maximum number of iterations.")

        return self.indices_to_domain(self._s + self._cm, self.domain_matrix)

    def tell(self, parameters: Matrix, objective_values: Vector):
        """Set the population and objective values. Note that parameters should match
        the parameters obtained by the prior ask() call. If it doesn't match (which
        might happen if a repair have been applied), then an exception will be raised.

        This function also updates best seen solutions with their objective values.

        Args:
            parameters: Matrix of parameter values representing the population.
            objective_values: Vector of objective values.
        Raises:
            ValueError: If parameters do not match with the ones obtained in the ask()
            call. This algorithm does not support such behavior since it keeps internal
            state that tracks the local search.
        """
        super(HillClimbingSolver, self).tell(parameters, objective_values)
        parameters, objective_values = self._process_told_parameters_and_objectives(
            parameters, objective_values
        )

        if (
            self.indices_to_domain(self._s + self._cm, self.domain_matrix) != parameters
        ).any():
            raise ValueError(
                "Modifying parameters is not supported by this algorithm. Please, make "
                "sure that you do not modify parameters between ask and tell."
            )
        self._incorporate_results_for_current_move(objective_values)
        self._transition_current_move()
        self._iterations_left -= 1

    def stop(self) -> bool:
        """Determine if the solver should terminate or not.

        Returns:
            Bool, True is the solver should terminate.
        """
        return self._iterations_left <= 0

    def best(self) -> Tuple[np.ndarray, float]:
        """Get the best solution.

        Returns:
            Tuple, (np.ndarray, float), the best solution and its objective value.
        """
        best_idx = np.argmin(self._best_obj_values)
        return self.parameters[best_idx], self.objective_values[best_idx]

    @property
    def _internal_objective_values(self) -> Vector:
        """Get the best, not sense-checked objective values.

        Returns:
            Vector of objective values.
        """
        return self._best_obj_values

    @property
    def parameters(self) -> Matrix:
        """Get the best parameters matrix seen.

        Returns:
            Matrix of parameters.
        """
        return self.indices_to_domain(self._best_s, self.domain_matrix)
