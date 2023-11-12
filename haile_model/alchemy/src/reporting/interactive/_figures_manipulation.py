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

import typing as tp

T = tp.TypeVar("T")  # noqa: WPS111
TDictValue = tp.Union[T, tp.Iterable[T], tp.Dict[str, T], "TDictValue"]
TDictKey = tp.Union[int, str]
TNestedDict = tp.Dict[TDictKey, TDictValue]


def flatten_dict(objects_tree: tp.Dict[TDictKey, T], separator: str) -> tp.Dict[str, T]:
    """
    Flattens a nested dictionary of objects into a single level dict.
    Each new key is formed like all preceding keys joined with ``separator``.

    Examples::

        >>> sample_figure = object()
        >>> figures = {
        ...     "a": sample_figure,
        ...     "b": {
        ...         "c": sample_figure,
        ...         "d": {"f": sample_figure, "g": sample_figure},
        ...         "a": sample_figure,
        ...     },
        ...     "": {"a": sample_figure, "": {"a": sample_figure}},
        ...     "c": {"a": [sample_figure, sample_figure]},
        ... }

        >>> flatten_dict(figures, separator=".")
        ... {
        ...     "a": sample_figure,
        ...     "b.c": sample_figure,
        ...     "b.d.f": sample_figure,
        ...     "b.d.g": sample_figure,
        ...     "b.a": sample_figure,
        ...     ".a": sample_figure,
        ...     "..a": sample_figure,
        ...     "c.a.0": sample_figure,
        ...     "c.a.1": sample_figure,
        ... }

    """
    flat_figures = [
        (key, fig) for key, fig in _get_key_figure(None, objects_tree, separator)
    ]
    return dict(flat_figures)


def _get_key_figure(
    prefix: tp.Optional[str],
    tree: TDictValue,
    separator: str,
) -> tp.Iterable[tp.Tuple[str, T]]:
    """
    Yields objects with flat keys from ``figures`` in dfs manner.
    Flat key is all the keys used to get to an ``object`` joined through ``separator``.

    Args:
        prefix: preceding key that led to this ``tree``
        tree: nested dict that might contain either object or list/subtree of objects
        separator: the resulting key for each node in ``tree`` is formed as
            ``f"{prefix}{separator}{key}"``

    Yields:
        tuple with key and object
    """
    for key, fig_or_dict in tree.items():
        key_with_prefix = f"{prefix}{separator}{key}" if prefix is not None else key
        if isinstance(fig_or_dict, (list, tuple)):
            fig_or_dict = dict(enumerate(fig_or_dict))
        if isinstance(fig_or_dict, dict):
            yield from _get_key_figure(key_with_prefix, fig_or_dict, separator)
        else:
            yield key_with_prefix, fig_or_dict
