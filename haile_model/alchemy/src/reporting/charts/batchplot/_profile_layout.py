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

import math
import typing as tp

import numpy as np

_MAX_SENSORS_PER_SUBPLOT = 4


class _XAxisGrid(object):
    def __init__(self, n_plots: int, horizontal_offset: float) -> None:
        """
        Implements grid for x-axis

        Args:
            horizontal_offset: is used for separating secondary axes from border
                and then to separate plot domain from secondary axis

        Notes:
            We create a split between objects on the x-axis.

            Plot (including all annotations, ticks and whatever) is constrained
            by its borders; all object lie within `left_border` and `right_border`.

            We use small offset from those borders for placing additional axes,
            those locations correspond to `left_border_plus_offset` and
                `right_border_plus_offset`.

            Traces are located inside the smaller region inside those borders
            from `left_domain` to `right_domain`.
        """
        self._offset = horizontal_offset

        self.left_border = float(0)
        self.right_border = 1.0

        n_plots_per_left_side = math.ceil(n_plots / 2)
        self.left_domain = self.left_border + (n_plots_per_left_side - 1) * self._offset
        n_plots_per_right_side = math.floor(n_plots / 2)
        self.right_domain = (
            self.right_border - (n_plots_per_right_side - 1) * self._offset
        )

    def get_axis_location(self, plot_id: int) -> tp.Dict[str, tp.Union[str, float]]:
        offset = (plot_id // 2) * self._offset
        if plot_id % 2 == 0:
            return dict(side="left", position=self.left_domain - offset)  # noqa: C408
        return dict(side="right", position=self.right_domain + offset)  # noqa: C408


class _YAxisGrid(object):
    def __init__(self, n_subplots: int, vertical_gap_between_subplots: float) -> None:
        self.y_domains = self._get_y_domains(n_subplots, vertical_gap_between_subplots)

    @staticmethod
    def _get_y_domains(
        n_subplots: int,
        vertical_gap_between_subplots: float,
    ) -> tp.List[tp.Tuple[float, float]]:
        """
        Returns the domain of y axes for subplots in a plotly figure
        (plots in subplots are going in top-bottom order, this will be accounted).

        Note:
            the first subplot will start from `vertical_gap_between_subplots`,
            it helps to create more space between plot & title
        """

        if n_subplots < 1:
            raise ValueError("This function works for subplots >= 1")

        if n_subplots == 1:
            # first subplot will start from `vertical_gap_between_subplots`,
            # it helps to create more space between plot & title
            return [(float(0), 1.0 - vertical_gap_between_subplots)]

        steps = np.linspace(0, 1, n_subplots + 1)
        # same here with first subplot (the latest)
        y_domain = list(zip(steps[:-1], steps[1:] - vertical_gap_between_subplots))
        y_domain.reverse()  # since we are building our subplots top down
        return y_domain


class _AxesReferences(object):
    def __init__(self, n_plots: int, subplot_id: int) -> None:
        """
        Create axes configurations for subplot in a plotly figure

        Args:
            subplot_id: index of the subplot in a plotly figure object

        Notes:
            All anchors and axes are indexed starting from 1 (plotly requirement)

        Returns: (x & y) axes names and (x & y) anchor names
        """

        x_axis_id = subplot_id + 1
        self.x_axis = f"xaxis{x_axis_id}"

        y_first_plot_id = subplot_id * _MAX_SENSORS_PER_SUBPLOT + 1
        y_axis_ids = [y_first_plot_id + plot_id for plot_id in range(n_plots)]
        self.y_axes = [f"yaxis{y_axis_id}" for y_axis_id in y_axis_ids]

        self.anchor_x = f"x{x_axis_id}"
        self.anchor_y = f"y{y_first_plot_id}"


class _TracesLocations(object):
    def __init__(self, sensors_to_plot_by_phases: tp.Dict[str, tp.List[str]]) -> None:
        """Object for mapping phase_id, sensor_id into plotly's x-axis and y-axis"""
        subplots_per_phase = np.cumsum(
            [
                math.ceil(len(sensors_to_plot) / _MAX_SENSORS_PER_SUBPLOT)
                for sensors_to_plot in sensors_to_plot_by_phases.values()
            ],
        )
        # cast to py `int`
        subplots_per_phase = [int(count) for count in subplots_per_phase]
        self._subplots_drawn_before_phase = [0, *subplots_per_phase[:-1]]

    def __getitem__(self, trace_id: tp.Tuple[int, int]) -> tp.Tuple[int, int]:
        phase_id, sensor_id = trace_id
        subplots_before = self._subplots_drawn_before_phase[phase_id]
        return (
            sensor_id // _MAX_SENSORS_PER_SUBPLOT + subplots_before + 1,
            sensor_id + subplots_before * _MAX_SENSORS_PER_SUBPLOT + 1,
        )


def get_profile_composition_layout(
    sensors_to_plot_by_phases: tp.Dict[str, tp.List[str]],
    vertical_gap_between_subplots: float = 0.05,
    offset_between_y_axes: float = 0.03,
) -> tp.Tuple[tp.Dict[str, tp.Dict], _TracesLocations]:
    """
    Create layout configurations for a single batch profile chart.
    This includes only compositional specifications like number of subplots and
    dependencies between axes.
    """
    total_subplots = sum(
        math.ceil(len(sensors) / _MAX_SENSORS_PER_SUBPLOT)
        for sensors in sensors_to_plot_by_phases.values()
    )

    subplots_height_ratios = _YAxisGrid(total_subplots, vertical_gap_between_subplots)

    # we assume it's always max traces per subplot to align appearance among subplots
    plots_per_subplot = _MAX_SENSORS_PER_SUBPLOT
    layout_params = {}
    for subplot_idx, y_domain in enumerate(subplots_height_ratios.y_domains):
        sub_layout_param = _get_single_layout(
            _AxesReferences(plots_per_subplot, subplot_idx),
            _XAxisGrid(plots_per_subplot, offset_between_y_axes),
            y_domain,
        )
        layout_params.update(sub_layout_param)

    return layout_params, _TracesLocations(sensors_to_plot_by_phases)


def _get_single_layout(
    axes_references: _AxesReferences,
    width_ratios: _XAxisGrid,
    height_ratio: tp.Tuple[float, float],
) -> tp.Dict[str, tp.Dict]:
    """Create layout configuration for multiple y-axis subplots.

    Notes:
        First line is shown on the left y-axis, next on the right, third on the left and
         the last on the right.
        To create such layout we use `width_ratios` which contains plot annotation's
         x-locations.

    Args:
        axes_references: contains axes names to reference
        width_ratios: domain of x-axis
        height_ratio: domains of y-axes

    Returns: layout parameters for plotly go.Figure object
    """

    return {
        axes_references.x_axis: dict(  # noqa: C408
            domain=[width_ratios.left_domain, width_ratios.right_domain],
            anchor=axes_references.anchor_y,
            matches="x",
        ),
        **{
            y_axis: dict(
                anchor="free",
                overlaying=axes_references.anchor_y,
                **width_ratios.get_axis_location(positional_id),
            )
            for positional_id, y_axis in enumerate(axes_references.y_axes)
        },
        # overwrite first y-axis; there's no location for the first trace because
        # plotly prettifies the span between the plot and its left axis
        axes_references.y_axes[0]: dict(  # noqa: C408
            anchor=axes_references.anchor_x,
            domain=list(height_ratio),
        ),
    }
