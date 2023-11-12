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
from collections import Counter
from numbers import Real
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy._lib._util import check_random_state
from six import string_types

from optimizer.domain import RealDimension
from optimizer.exceptions import MaxIterationError
from optimizer.solvers.continuous.base import ScaledContinuousSolver
from optimizer.types import Matrix, Vector
from optimizer.utils.initializer import latin_hypercube, uniform_random


class AdaptiveParameter:
    def value(self) -> float:
        """Obtain the current value of the parameter

        Returns: float, the current value
        """
        raise NotImplementedError("Override this method")

    def give_feedback(self, previous_objective: np.ndarray, new_objective: np.ndarray):
        """
        Receives the feedback provided by the user of this parameter. This method
            processes the feedback for tuning the logic of how the next value() call
            will generate the current value of the parameter.

        Args:
            previous_objective: values of the current solutions that the optimizer has.
            new_objective: values of the trial solutions that the optimizer obtained
                after using the current value of the parameter.
        """
        if previous_objective.shape != new_objective.shape:
            raise ValueError("Invalid objectives shapes")
        self._process_feedback(previous_objective, new_objective)

    def _process_feedback(
        self, previous_objective: np.ndarray, new_objective: np.ndarray
    ):
        """Override this method to implement a concrete tuning algorithm."""
        raise NotImplementedError("Override this method")


class ConstantAdaptiveParameter(AdaptiveParameter):
    """Implements a constant parameter."""

    def __init__(self, value: float):
        super().__init__()
        self._value = value

    def value(self) -> float:
        return self._value

    def _process_feedback(
        self, previous_objective: np.ndarray, new_objective: np.ndarray
    ):
        pass


class NormalNonAdaptiveParameter(AdaptiveParameter):
    """Implements a sampled parameter from a normal distribution with mu and sigma.
    Ignores feedback since these parameters are fixed. Each call to value() will
    generate a new sample.
    """

    def __init__(self, mu: float, sigma: float, seed=None):
        super().__init__()
        self._mu = mu
        self._sigma = sigma
        self._rng = check_random_state(seed)

    def value(self) -> float:
        return self._rng.normal(self._mu, self._sigma)

    def _process_feedback(
        self, previous_objective: np.ndarray, new_objective: np.ndarray
    ):
        pass


class NormalAdaptiveParameter(AdaptiveParameter):
    """Implements a sampled parameter from a normal distribution with mu and sigma."""

    def __init__(self, mu: float, sigma: float, n: float = 1, seed=None):
        """
        Construct your parameter which follows a normal distribution.
            Over-time, the mean will be updated, if using 1 value per
            population member.

        Args:
            mu: Mean of normal distribution (will be updated)
            sigma: Standard Deviation of normal distribution
            n: Size of population/number of draws to take from N(mu, std)
            seed: Random number seed
        """
        super().__init__()
        self._sigma = sigma
        self._n = n
        self._rng = check_random_state(seed)

        self._mu = self._rng.normal(mu, self._sigma, size=(self._n, 1))
        self._mu_success = np.empty(0)

    def value(self) -> float:
        return self._mu

    def _process_feedback(
        self, previous_objective: np.ndarray, new_objective: np.ndarray
    ):
        """Process feedback to incorporate changes in mutator choice"""
        improved = new_objective < previous_objective
        self._mu_success = np.append(self._mu_success, self._mu[improved, :])
        if np.sum(improved) > 0:
            mu_new = np.mean(self._mu_success)
            self._mu = self._rng.normal(mu_new, self._sigma, size=(self._n, 1))
            self.reset_tracker()

    def reset_tracker(self):
        self._mu_success = np.empty(0)


class Mutator:
    def __init__(self, f: Union[float, AdaptiveParameter], seed=None):
        """Constructor.

        Args:
            f: A bigger f implies a more aggressive exploration. f should be between 0
               and 2.
            seed: random seed
        """
        # If down the road we experiment with Mutators that do not have a f parameter,
        # then it is a good idea to refactor these classes, moving the f parameter to
        # subclass __init__ functions. At the implementation time of this code we only
        # considered mutations that have make use of the parameter f, that's why we've
        # left it in the parent class.
        self._f = ConstantAdaptiveParameter(f) if isinstance(f, (float, int)) else f
        self._rng = check_random_state(seed)

    def give_feedback(self, previous_objective: np.ndarray, new_objective: np.ndarray):
        """Forwards feedback to adaptive parameter f."""
        if previous_objective.shape != new_objective.shape:
            raise ValueError("Invalid objectives shapes")
        self._f.give_feedback(previous_objective, new_objective)

    def __call__(self, solutions: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Need to override Mutator.__call__")

    def _mutation_parent_indexes(self, n: int, k: int) -> np.ndarray:
        """Returns a n x k matrix, where n is number of solutions and k is a parameter.
        Each position is a value between 0 and n-1 (used for indexing the solutions).
        The i-th row can have values 0...i-1,i+1...n-1
        Finally, each row does not contain duplicated values.
        """
        if k > n - 1:
            raise ValueError("k has to be at most n-1, where n is row count")

        # It's possible that a candidates approach isn't the efficient one.
        # To be revised. It might be cheaper to fix the idx row by row.
        candidates = np.arange(n - 1)
        candidates = np.tile(candidates, (n, 1))
        candidates[np.arange(n - 1), [np.arange(n - 1)]] = n - 1
        # As an example, candidates with n=6 is the following matrix.
        # This is used for sampling, so we don't really care about the order
        #     [[5 1 2 3 4]
        #      [0 5 2 3 4]
        #      [0 1 5 3 4]
        #      [0 1 2 5 4]
        #      [0 1 2 3 5]
        #      [0 1 2 3 4]]
        #     print(candidates)

        # This creates an nxn random matrix and then sorts it. This is unlikely to be
        # the best option. But it is faster than calling np.random.permutation on each
        # candidates row. The 10000 number is to avoid having a strong bias towards
        # solutions with lower indexes.
        idx = np.argsort(self._rng.randint(0, 10000 * k, (n, n - 1)))[:, :k]
        out = candidates[np.arange(n)[:, np.newaxis], idx]
        return out


class Rand1Mutator(Mutator):
    """Implements DE/rand/1 mutator"""

    def __call__(self, solutions: np.ndarray, fitness: np.ndarray):
        f = self._f.value()
        n = solutions.shape[0]
        idx = self._mutation_parent_indexes(n, 3)

        mutations = solutions[idx[:, 0]] + f * (
            solutions[idx[:, 1]] - solutions[idx[:, 2]]
        )
        return mutations


class Rand2Mutator(Mutator):
    """Implements DE/rand/2 mutator"""

    def __call__(self, solutions: np.ndarray, fitness: np.ndarray):
        f = self._f.value()
        n = solutions.shape[0]
        idx = self._mutation_parent_indexes(n, 5)

        mutations = solutions[idx[:, 0]] + f * (
            (solutions[idx[:, 1]] - solutions[idx[:, 2]])
            + (solutions[idx[:, 3]] - solutions[idx[:, 4]])
        )
        return mutations


class Best1Mutator(Mutator):
    """Implements DE/best/1 mutator"""

    def __call__(self, solutions: np.ndarray, fitness: np.ndarray):
        idx_best = np.argmin(fitness)
        n = solutions.shape[0]
        f = self._f.value()
        idx = self._mutation_parent_indexes(n, 2)

        mutations = solutions[idx_best] + f * (
            solutions[idx[:, 0]] - solutions[idx[:, 1]]
        )
        return mutations


class Best2Mutator(Mutator):
    """Implements DE/best/2 mutator"""

    def __call__(self, solutions: np.ndarray, fitness: np.ndarray):
        idx_best = np.argmin(fitness)
        n = solutions.shape[0]
        f = self._f.value()
        idx = self._mutation_parent_indexes(n, 4)

        mutations = solutions[idx_best] + f * (
            (solutions[idx[:, 0]] - solutions[idx[:, 1]])
            + (solutions[idx[:, 2]] - solutions[idx[:, 3]])
        )
        return mutations


class CurrentToBest1Mutator(Mutator):
    """Implements DE/current to best/1 mutator"""

    def __call__(self, solutions: np.ndarray, fitness: np.ndarray):
        idx_best = np.argmin(fitness)
        n = solutions.shape[0]
        f = self._f.value()
        idx = self._mutation_parent_indexes(n, 2)

        mutations = solutions + f * (
            (solutions[idx_best] - solutions)
            + (solutions[idx[:, 0]] - solutions[idx[:, 1]])
        )
        return mutations


class CompositeMutator(Mutator):
    """Meta-mutator, which applies different mutators to each potential solution"""

    def __init__(
        self,
        f: Union[float, AdaptiveParameter],
        mutators: Dict[str, Mutator],
        learning_frequency: int = 10,
        learning_interval: int = 60,
        seed=None,
    ):
        """
        The CompositeMutator is a meta-mutator, in that it adapts which choice of
            mutator to apply to each potential solution (e.g. row of parameters).

        This mutator maintains a probability distribution over the user-supplied
            dictionary of mutators. Over the course of the `learning_interval`, this
            pdf is updated according to the proportion of how many trial/mutant vectors
            are created which succeed in improving the objective function.

        Specifically, the learning is done every `learning_frequency` iterations,
            and is not done after `learning_interval` iterations. By making this second
            parameter large, users can effectively keep adaptation on for the entire
            optimization.

        Args:
            f: Used here for backwards-compatibility. Each mutator in `mutators`
                argument will have it's own value of F.
            mutators: A dictionary of mutators, keyed by a string identifying the
                strategy.
            learning_frequency: How often to update the probability of mutator
                choice.
            learning_interval: Length of time to update distribution over mutator
                choice.
            seed: Random number seed.
        """
        super().__init__(f)
        self._rng = check_random_state(seed)
        self._iter_count = 1
        if not mutators:
            raise ValueError(
                "The library of mutators supplied here is empty. "
                "Please pass at least 1 mutator in the `mutators`"
                " argument"
            )
        self._mutators = mutators
        self._mutator_names = list(mutators)

        self._success_probs = {
            name: 1.0 / len(self._mutators) for name in self._mutators
        }

        self._iter_count = 1
        self._learning_frequency = learning_frequency
        self._learning_interval = learning_interval

        self._pop_success = Counter()
        self._pop_fail = Counter()
        self._mutator_choice = np.empty(0)

    def __call__(self, solutions: np.ndarray, fitness: np.ndarray):
        # Adaption and "learning"
        if (
            self._iter_count > 1 and self._iter_count < self._learning_frequency
        ) and self._iter_count % self._learning_frequency == 0:
            # Generalized SaDE to larger "library" of mutation operators
            probs = {
                name: self._pop_success[name]
                / (self._pop_success[name] + self._pop_fail[name])
                for name in self._mutator_names
            }
            self._success_probs = {
                name: probs[name] / sum(list(probs.values()))
                for name in self._mutator_names
            }
            self.reset_counters()

        mutator_probabilities = [
            self._success_probs[mutator_] for mutator_ in self._mutator_names
        ]

        self._mutator_choice = self._rng.choice(
            self._mutator_names, size=len(solutions), p=mutator_probabilities
        )

        mutant = np.zeros(solutions.shape)
        mutated_solutions = {
            mutator_name: mutator(solutions, fitness)
            for mutator_name, mutator in self._mutators.items()
        }
        # Loop over mutations. Each mutator is vectorized and with
        # simple operations, so quite quick within loop
        for mutator_name in self._mutators:
            idx_mask = np.argwhere(self._mutator_choice.flatten() == mutator_name)
            mutant[idx_mask] = mutated_solutions[mutator_name][idx_mask, :]
        return mutant

    def reset_counters(self):
        self._pop_success = Counter()
        self._pop_fail = Counter()

    def give_feedback(self, previous_objective: np.ndarray, new_objective: np.ndarray):
        """Overriding parent method.
        Forwarding feedback to "f" parameter and processing feedback for mutator choice
        """
        if previous_objective.shape != new_objective.shape:
            raise ValueError("Invalid objectives shapes")
        self._f.give_feedback(previous_objective, new_objective)

        # if feedback is given but mutator_choice not made first:
        if self._mutator_choice.size == 0:
            self._mutator_choice = self._rng.choice(
                list(self._mutators.keys()), size=len(previous_objective)
            )
        self._process_feedback(self._mutator_choice, previous_objective, new_objective)

    def _process_feedback(
        self,
        value: np.ndarray,
        previous_objective: np.ndarray,
        new_objective: np.ndarray,
    ):
        """Process feedback to incorporate changes in mutator choice"""
        improved = new_objective < previous_objective
        self._pop_success += Counter(value[improved])
        self._pop_fail += Counter(value[~improved])
        self._iter_count += 1


class Crossover:
    """Interface for crossover implementations."""

    def __init__(self, cr: Union[float, AdaptiveParameter], seed=None):
        """Constructor.

        Args:
            cr: crossover random threshold. Used in randomized approaches to combine
                existing solutions and mutated solutions. Typically it captures how much
                of the mutant and how much of the current goes into the trial solution.
                Different concrete implementations might have different usages for this
                parameter.
                We kept this name since it's the typically used name in the literature.
            seed: Random seed.
        """
        # If down the road we experiment with Crossover functions that do not have a cr
        # parameter, then it is a good idea to refactor these classes, moving the cr to
        # subclass __init__ functions. At the implementation time of this code we only
        # considered crossovers that have the single parameter CR, that's why we have it
        # in the parent class.
        self._cr = ConstantAdaptiveParameter(cr) if isinstance(cr, (float, int)) else cr
        self._rng = check_random_state(seed)

    def __call__(self, solutions: np.ndarray, mutants: np.ndarray) -> np.ndarray:
        """Combines solutions with their corresponding mutants.

        Args:
            solutions: Current solutions (also called target variables or current
                population in the literature)
            mutants: Mutations corresponding to each solution. These are ty
        """
        if solutions.shape != mutants.shape:
            raise ValueError("Solutions and mutants must have the same shape")
        return self._combine(solutions, mutants)

    def give_feedback(self, previous_objective: np.ndarray, new_objective: np.ndarray):
        """Forwards feedback to adaptive parameter cr."""
        self._cr.give_feedback(previous_objective, new_objective)

    def _combine(self, solutions: np.ndarray, mutants: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Need to override")


class BinomialCrossover(Crossover):
    """Binomial crossover implementation.

    For each pair (solution, mutant), the combination works as follows:

        - Compute a random value in [0,1] for each dimension j of the solution, let's
            call value r_j.
        - Take one random dimension k
        - Define each dimension j of the combined solution as:
          - out_j = mutant_j if (r_j < cr) or (j=k)
          - out_j = solution_j otherwise

    The purpose of the special dimension k is to enforce that the newly generated
    solution will differ in at least something with respect to the current solution.

    Keep in mind that these generated offsprings are trials, and if they improve the
    objective value we then update the current solutions.
    """

    def _generate_forced_crossover_positions(self, n: int, k: int) -> np.ndarray:
        """Returns a (n x k) boolean matrix, where each row has exactly one
        True value.
        """
        is_ = np.arange(n)
        js_ = self._rng.randint(0, k, size=n)
        z = np.zeros((n, k)) != 0
        z[is_, js_] = True
        return z

    def _combine(self, solutions: np.ndarray, mutants: np.ndarray) -> np.ndarray:
        mut = self._generate_forced_crossover_positions(*solutions.shape)
        rand = self._rng.random(solutions.shape)
        crossover_mask = (rand < self._cr.value()) | mut
        return np.where(crossover_mask, mutants, solutions)


class ExponentialCrossover(Crossover):
    """Exponential crossover implementation.

    For each pair (solution, mutant), the combination works as follows:

        - Compute a random value in [0,1] for each dimension j of the solution, let's
            call value r_j.
        - Compute how many consecutive values of r (r_j, r_j+1, ...) are < cr. Call
            this number L
        - Take one random dimension k
        - "Crossover" the continuous section of the mutant, from r_j to r_j + L.
          - out_j: out_j+L = mutant_j: mutant_j+L
          - out_j = solution_j otherwise

    This crossover operation takes advantage when features are "nearby" to each other.
    Users can consider variants such as Shuffled Exponential Crossover when this is
    not the case.
    Keep in mind that these generated offsprings are trials, and if they improve the
    objective value we then update the current solutions.
    """

    def _generate_forced_crossover_positions(self, n: int, k: int) -> np.ndarray:
        """Returns a (n x k) boolean matrix, where each row has exactly one
        True value.
        """
        is_ = np.arange(n)
        js_ = self._rng.randint(0, k, size=n)
        z = np.zeros((n, k)) != 0
        z[is_, js_] = True
        return z

    def _combine(self, solutions: np.ndarray, mutants: np.ndarray) -> np.ndarray:
        # solutions.shape = (n, D)
        solutions_shape = solutions.shape
        rand = self._rng.random(solutions_shape)
        lengths = np.argmin(rand < self._cr.value(), axis=1)

        # edge case of whole "genome" when all entries are true
        lengths[
            np.argwhere(np.sum(rand < self._cr.value(), axis=1) == solutions_shape[1])
        ] = solutions_shape[1]

        # Find where the random crossover positions are
        starting_points = np.argmax(
            self._generate_forced_crossover_positions(*solutions_shape), axis=1
        )
        # Generate coords tuples where crossover happens and convert to array.
        coord_array = np.array(
            [
                (idx, y % solutions_shape[1])
                for idx, (start, length) in enumerate(zip(starting_points, lengths))
                for y in range(start, start + length)
            ],
            dtype=int,
        )
        end_mask = np.zeros_like(mutants)
        # If there is no crossover, we skip and return
        if coord_array.shape[0] != 0:  # pylint: disable=E1136
            x = coord_array[:, 0]
            y = coord_array[:, 1]
            end_mask[x, y] = 1.0
        return np.where(end_mask, mutants, solutions)


class AdaptiveStrategy:
    def __init__(self, strategies: Dict[str, Union[Crossover, Mutator]]):
        self._strategies = strategies

    def get_strategy(self) -> Tuple[str, Union[Crossover, Mutator]]:
        """Obtain the strategy as suggested by the adaptive method. This can be a non
        deterministic function.

        Returns: Tuple, with the name of the strategy and the strategy itself.

        """
        raise NotImplementedError("Override this method")

    def give_feedback(
        self, strategy: str, previous_objective: np.ndarray, new_objective: np.ndarray
    ):
        """
        Receives the feedback provided by the optimization engine. This does two main
            things:

        - Forwards the feedback to the picked strategy. The typical use case would be
            that this strategy uses this feedback to adjust some adaptive parameter.
        - Processes the feedback for tuning of the AdaptiveStrategy by calling an
            overridden method.

        Args:
            strategy: str, indicating the name of the strategy picked. It has to be a
                value in the keys of strategies dictionary given in the constructor.

            previous_objective: values of the current solutions that the optimizer has.
            new_objective: values of the trial solutions that the optimizer obtained
                after applying the strategy.
        """
        if previous_objective.shape != new_objective.shape:
            raise ValueError("Invalid objectives shapes")
        self._strategies[strategy].give_feedback(previous_objective, new_objective)
        self._process_feedback(strategy, previous_objective, new_objective)

    def _process_feedback(
        self, strategy: str, previous_objective: np.ndarray, new_objective: np.ndarray
    ):
        """
        Override this method to implement the adaptive strategy updates to internal
            states. These internal states might need an override of __init__ and they
            are expected to be used to generate better strategy choices in the next
            call to self.get_strategy().

        Args:
            strategy: str, indicating the name of the strategy picked. It has to be a
                value in the keys of strategies dictionary given in the constructor.
            previous_objective: values of the current solutions that the optimizer has.
            new_objective: values of the trial solutions that the optimizer obtained
                after applying the strategy.
        """
        raise NotImplementedError("Override this method")


class RandomAdaptiveStrategy(AdaptiveStrategy):
    def __init__(self, strategies: Dict[str, Union[Crossover, Mutator]], seed=None):
        super().__init__(strategies)
        self._rng = check_random_state(seed)

    def get_strategy(self) -> Tuple[str, Union[Crossover, Mutator]]:
        strategy_picked = self._rng.choice(list(self._strategies))
        return strategy_picked, self._strategies[strategy_picked]

    def _process_feedback(
        self, strategy: str, previous_objective: np.ndarray, new_objective: np.ndarray
    ):
        pass


class AdaptiveBinomialCrossOver(AdaptiveStrategy):
    """
    Adaptive Strategy for updating cross-over rate CR over each iteration.
    """

    def __init__(self, strategies: Dict[str, Union[Crossover, Mutator]], seed=None):
        super().__init__(strategies)
        self._rng = check_random_state(seed)

    def get_strategy(self) -> Tuple[str, Union[Crossover, Mutator]]:
        strategy_picked = self._rng.choice(list(self._strategies))
        return strategy_picked, self._strategies[strategy_picked]

    def _process_feedback(
        self, strategy: str, previous_objective: np.ndarray, new_objective: np.ndarray
    ):
        self._strategies[strategy].give_feedback(previous_objective, new_objective)


class AdaptiveDifferentialEvolutionSolver(ScaledContinuousSolver):
    def __init__(
        self,
        domain: List[Union[Tuple[Real, Real], RealDimension]],
        sense: str = "minimize",
        seed: int = None,
        popsize: int = 10,
        maxiter: int = 100,
        init: Union[str, Matrix] = "latinhypercube",
        mutator_strategy_picker: AdaptiveStrategy = None,
        crossover_strategy_picker: AdaptiveStrategy = None,
    ):
        super().__init__(domain, sense=sense, seed=seed)

        self._rng = check_random_state(seed)
        self._popsize = popsize
        self._maxiter = maxiter
        self._iterations_left = maxiter

        self._solutions = self._init_population(init)
        self._obj_values = np.inf * np.ones(self._popsize)

        self._mutator_picker = self._init_mutators(mutator_strategy_picker)
        self._crossover_picker = self._init_crossovers(crossover_strategy_picker)

        # Holds the current ask/tell mutation and crossover strategy.
        # Used for giving feedback to adaptive strategy pickers
        self._current_mutation_strategy = None
        self._current_crossover_strategy = None

    def _init_crossovers(
        self, crossover_adaptive_strategy_picker: AdaptiveStrategy
    ) -> AdaptiveStrategy:
        if not crossover_adaptive_strategy_picker:
            crossover_adaptive_strategy_picker = RandomAdaptiveStrategy(
                self._default_crossover_strategies()
            )
        return crossover_adaptive_strategy_picker

    def _init_mutators(
        self, mutator_adaptive_strategy_picker: AdaptiveStrategy
    ) -> AdaptiveStrategy:
        """

        Args:
            mutator_adaptive_strategy_picker:

        Returns:

        """
        if not mutator_adaptive_strategy_picker:
            mutator_adaptive_strategy_picker = RandomAdaptiveStrategy(
                self._default_mutation_strategies()
            )
        return mutator_adaptive_strategy_picker

    def _default_mutation_strategies(self):
        """
        Default mutation dictionary is kept here. Each element in this dictionary
            will be among the pool of mutator options available to the mutator picker.

        Returns: The dict parameter for constructing the AdaptiveStrategy.
        """
        f = ConstantAdaptiveParameter(1)
        return {
            "Rand1Mutator": Rand1Mutator(f=f, seed=self._rng),
            "Best1Mutator": Best1Mutator(f=f, seed=self._rng),
            "CurrentToBest1Mutator": CurrentToBest1Mutator(f=f, seed=self._rng),
        }

    def _default_crossover_strategies(self):
        """
        Default crossover dictionary is kept here. Each element in this dictionary
            will be among the pool of options available to the crossover picker.

        Returns: The dict parameter for constructing the AdaptiveStrategy.
        """
        return {"BinomialCrossover": BinomialCrossover(cr=0.7, seed=self._rng)}

    def _init_population(self, init: Union[str, Matrix]):
        if isinstance(init, string_types):
            if init == "latinhypercube":
                solutions = latin_hypercube(
                    (self._popsize, len(self._domain)), self.rng
                )
            elif init == "random":
                solutions = uniform_random((self._popsize, len(self._domain)), self.rng)
            else:
                raise ValueError(f"Initialization function {init} not implemented.")
        else:
            if self._popsize != init.shape[0]:
                raise ValueError("Popsize does not match with initializing matrix.")
            solutions = self._init_population_array(init)
        return solutions

    def _mutate(self) -> np.ndarray:
        """
        Constructs a set of mutants for later crossover with the current solutions.
            This method picks a strategy according to the adaptive picker and calls it.

        Returns: np.ndarray of same shape as self._solutions. These are the individuals
            that will be blended with the current solutions to form up the trials.
        """
        mutator_name, mutator = self._mutator_picker.get_strategy()
        mutants = mutator(self._solutions, self._obj_values)
        self._current_mutation_strategy = mutator_name
        return mutants

    def _crossover(self, mutants: np.ndarray) -> np.ndarray:
        """
        Given the mutants for each solution, perform the crossover by picking the
            adaptive crossover logic that corresponds.

        Args:
            mutants: np.ndarray of shape self._solutions.shape

        Returns: np.ndarray containing all trials, these are the solutions that the user
            will evaluate next. Same shape as self._solutions.shape

        """
        crossover_name, crossover = self._crossover_picker.get_strategy()
        trials = crossover(self._solutions, mutants)
        self._current_crossover_strategy = crossover_name
        return trials

    def ask(self) -> np.ndarray:
        """Get the current trial population to evaluate.

        Returns:
            np.array of current trial solutions.

        Raises:
            MaxIterationError: when called after maximum iterations.
        """
        if self._iterations_left <= 0:
            raise MaxIterationError("No more iterations left")
        self._iterations_left -= 1

        mutants = self._mutate()
        trials = self._crossover(mutants)
        trials = self._clip(trials)
        trials = self._inverse_transform_parameters(trials)

        return trials

    def stop(self) -> bool:
        """Determine if we should stop iterating the Ask and Tell loop.

        Returns:
            Boolean, True if we should stop.
        """
        return self._iterations_left <= 0

    def best(self) -> Tuple[np.ndarray, float]:
        """Get the best solution and its objective value.
        Internal sorting takes care of moving the best solution to the 0 index.

        Returns:
            Vector and float, the solution vector and its objective value.
        """
        best_idx = np.argmin(self._obj_values)
        return self.parameters[best_idx], self.objective_values[best_idx]

    def tell(self, parameters: Matrix, objective_values: Vector):
        """
        Set the population and objective values.
            Updates the internal population based on which individuals are
            performing better.

        Args:
            parameters: Matrix of parameter values representing the population.
            objective_values: Vector of objective values.
        """
        super().tell(parameters, objective_values)
        parameters, objective_values = self._process_told_parameters_and_objectives(
            parameters, objective_values
        )
        parameters = self._transform_parameters(parameters)
        assert (parameters == parameters.clip(0, 1)).all()

        self._give_feedback_to_mutation_picker(objective_values)
        self._give_feedback_to_crossover_picker(objective_values)

        self._update_current_trial(parameters, objective_values)

    def _give_feedback_to_mutation_picker(self, objective_values: np.ndarray):
        """Pass current feedback to the AdaptiveStrategy that picks the mutator

        Args:
            objective_values: Values of the objective function on the trial population
        """
        if self._current_mutation_strategy:
            self._mutator_picker.give_feedback(
                self._current_mutation_strategy, self.objective_values, objective_values
            )

        # Clear for safety, this is set in an ask() call.
        self._current_mutation_strategy = None

    def _give_feedback_to_crossover_picker(self, objective_values: np.ndarray):
        """Pass current feedback to the AdaptiveStrategy that picks the crossover

        Args:
            objective_values: Values of the objective function on the trial population
        """
        if self._current_crossover_strategy:
            self._crossover_picker.give_feedback(
                self._current_crossover_strategy,
                self.objective_values,
                objective_values,
            )

        # Clear for safety, this is set in an ask() call.
        self._current_crossover_strategy = None

    def _update_current_trial(
        self, parameters: np.ndarray, objective_values: np.ndarray
    ):
        """
        Updates self._solutions and self._obj_values based on objective values
            evaluated by the user.

        Args:
            parameters: the trial solutions that were just evaluated by the user
            objective_values: The objective values for these parameters
        """
        improved = objective_values < self._obj_values
        self._solutions[improved] = parameters[improved]
        self._obj_values[improved] = objective_values[improved]

    @property
    def _internal_objective_values(self) -> Vector:
        """Get the internal objective values.

        Returns:
            Vector of the current objective values.
        """
        return self._obj_values

    @property
    def parameters(self):
        """Get the current parameter values.

        Returns:
            Matrix of current parameters.
        """
        return self._inverse_transform_parameters(self._solutions)
