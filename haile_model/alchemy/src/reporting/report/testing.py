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

from __future__ import annotations

import typing as tp

from reporting.report.rendering import RENDERED_TYPES, TFigure, TFiguresDict

# Ok to ignore S101 here since the functions in this module are used in connection
#  with testing


# TODO: Check if list elements should be RENDERED_TYPES or if
#   just anything that can be rendered would still be ok
#  Ok to ignore WPS110 because here ``obj`` is a good name since the input is
# "some kind of object" that we want to know whether we can render or not
def assert_can_be_rendered(
    obj: tp.Union[TFigure, TFiguresDict],  # noqa: WPS110
    obj_is_top_level: bool = True,
    unpack_dict: bool = False,
) -> None:
    """
    Raises an Assertion error if the object cannot be rendered


    ``obj_is_top_level`` forces to unpack a dict because the first level
        must be unpackable, and shows this explicitly
    ``unpack_dict`` forces to unpack a dict because one wants to unpack a dict

    What is being checked:
    - first one makes sure that the first level has a ``.items`` method, and that the
        object to be tested can be unpacked into a name ``name`` and a value ``element``
    - then one iterates over the names and elements and checks
        - that the name exists
        - that the value
            - either is an instance of one of the ``RENDERED_TYPES``
            - or is a list of instances of one of the ``RENDERED_TYPES``
            - or is a dict that can be rendered (checked recursively)

    Original code was this:

    def assert_can_be_rendered(figs: tp.Union[TFigure, TFiguresDict]) -> None:
        for fig_name, element in figs.items():
            assert fig_name
            if isinstance(element, RENDERED_TYPES):
                pass
            elif isinstance(element, list):
                for fig in element:
                    assert isinstance(fig, RENDERED_TYPES)
            elif isinstance(element, dict):
                assert_can_be_rendered(element)
            else:
                raise AssertionError
    """
    if obj_is_top_level or unpack_dict:
        for name, element in obj.items():
            assert name  # noqa: S101
            assert_can_be_rendered(element, obj_is_top_level=False)
        return  # So after here everything is at levels below the top one
    if isinstance(obj, RENDERED_TYPES):
        return
    if isinstance(obj, list):
        for list_element in obj:
            assert isinstance(list_element, RENDERED_TYPES)  # noqa: S101
        return
    if isinstance(obj, dict):
        assert_can_be_rendered(obj, obj_is_top_level=False, unpack_dict=True)
        return
    raise AssertionError
