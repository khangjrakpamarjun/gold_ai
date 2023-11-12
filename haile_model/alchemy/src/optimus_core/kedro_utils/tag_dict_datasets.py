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
io classes for TagDict
"""
import typing as tp

from ..tag_dict.tag_dict import TagDict


class KedroNotInstalledError(Exception):
    """Raised when Kedro is not installed."""


try:  # noqa: WPS229
    from kedro.extras.datasets.pandas import CSVDataSet, ExcelDataSet
    from kedro.io.core import Version

except ImportError:
    raise KedroNotInstalledError(
        "Kedro must be installed in order to use `kedro_utils` module",
    )


_TAG_DICT_LOAD_ARGS = frozenset(("validate",))


class TagDictCSVLocalDataSet(CSVDataSet):
    """
    Loads and saves a TagDict object from/to csv
    This is an extension of the Kedro Pandas CSV Dataset
    """

    def __init__(
        self,
        filepath: str,
        load_args: tp.Dict[str, tp.Any] = None,
        save_args: tp.Dict[str, tp.Any] = None,
        version: Version = None,
        credentials: tp.Dict[str, tp.Any] = None,
        fs_args: tp.Dict[str, tp.Any] = None,
    ) -> None:
        self._td_load_args = _extract_tag_dict_specific_args(load_args)
        super().__init__(
            filepath=filepath,
            load_args=load_args,
            save_args=save_args,
            version=version,
            credentials=credentials,
            fs_args=fs_args,
        )

    def _load(self) -> TagDict:
        df = super()._load()
        return TagDict(df, **self._td_load_args)

    def _save(self, data: TagDict) -> None:
        df = data.to_frame()
        super()._save(df)


class TagDictExcelLocalDataSet(ExcelDataSet):
    """
    Loads and saves a TagDict object from/to excel

    This is an extension of the Kedro Text Dataset

    To load from a specific sheet, add "sheet_name" to the
    "load_args" in your catalog entry. To save to a specific
    sheet, add "sheet_name" to the "save_args" in your catalog entry.

    """

    def __init__(
        self,
        filepath: str,
        engine: str = "openpyxl",
        load_args: tp.Dict[str, tp.Any] = None,
        save_args: tp.Dict[str, tp.Any] = None,
        version: Version = None,
        credentials: tp.Dict[str, tp.Any] = None,
        fs_args: tp.Dict[str, tp.Any] = None,
    ) -> None:
        self._td_load_args = _extract_tag_dict_specific_args(load_args)
        super().__init__(
            filepath=filepath,
            engine=engine,
            load_args=load_args,
            save_args=save_args,
            version=version,
            credentials=credentials,
            fs_args=fs_args,
        )

    def _load(self) -> TagDict:
        df = super()._load()
        return TagDict(df, **self._td_load_args)

    def _save(self, data: TagDict) -> None:
        df = data.to_frame()
        super()._save(df)


def _extract_tag_dict_specific_args(
    load_args: tp.Dict[str, tp.Any],
) -> tp.Dict[str, tp.Any]:
    if load_args is None:
        return {}
    return {
        load_arg: load_args.pop(load_arg)
        for load_arg in _TAG_DICT_LOAD_ARGS
        if load_arg in load_args
    }
