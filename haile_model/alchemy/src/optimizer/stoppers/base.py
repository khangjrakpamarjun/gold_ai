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
Stoppers base module.
"""

import abc
from typing import List


class BaseStopper(abc.ABC):
    """
    Base class for stoppers.
    These classes offer an alternative way to stop the optimization.
    """

    _stop = False

    def stop(self) -> bool:
        """Stop getter.

        Returns:
            bool, True if the search should stop.
        """
        return self._stop

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        """Updates the internal stopping state.

        Args:
            *args: list of arguments to pass to a stopper.
            **kwargs: dictionary of keyword arguments to pass to a stopper.
        """


class MultiStopper:
    """
    Wrapper for any number of Stoppers.
    """

    def __init__(self, stoppers: List[BaseStopper]):
        """Constructor.

        Args:
            *stoppers: list of stoppers.

        Raises:
            ValueError: if the an element in stoppers is not an instance of BaseStopper.
        """
        for stopper in stoppers:
            if not isinstance(stopper, BaseStopper):
                raise ValueError(
                    f"Invalid object of type {type(stopper).__name__} provided."
                )

        self.stoppers = stoppers

    def __iter__(self):
        """Iterates over the stored list of stoppers.

        Returns:
            list_iterator.
        """
        return iter(self.stoppers)

    def stop(self) -> bool:
        """Returns True if any stored stoppers return True.

        Returns:
            bool, True if the search should stop.
        """
        return any(stopper.stop() for stopper in self)

    def update(self, *args, **kwargs):
        """Updates the internal stop state of all stored stoppers.

        Args:
            *args: list of arguements to pass to a solver.
            **kwargs: keyword arguments to pass to a solver.

        Returns:
            bool, True if the search should stop.
        """
        for stopper in self:
            stopper.update(*args, **kwargs)
