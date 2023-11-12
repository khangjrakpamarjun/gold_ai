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
Wall time based stopper.
"""

import time

from optimizer.stoppers.base import BaseStopper


class WallTimeStopper(BaseStopper):
    """
    This class will return True from `stop` if N minutes have overlapped since the
    first call to `update`.
    """

    def __init__(self, minutes: int):
        """Constructor.

        Args:
            minutes: number of minutes to wait before returning True in `stop`.

        Raises:
            ValueError: if `minutes` is negative.
        """
        if minutes < 0:
            raise ValueError(
                f"Provided value for minutes must be nonnegative. Provided {minutes}"
            )

        self.seconds = minutes * 60
        self.first_call = None

    def stop(self) -> bool:
        """Stop method override.

        Returns:
            True if enough time has passed or we have not yet called `update`.
        """
        return (
            self.first_call is not None and time.time() - self.first_call > self.seconds
        )

    def update(self, *args, **kwargs):  # pylint: disable=arguments-differ
        """Update the internal state based off how much time has elapsed.

        Args:
            *args: unused positional arguments.
            **kwargs: unused keyword arguments.
        """
        self.first_call = time.time() if self.first_call is None else self.first_call
