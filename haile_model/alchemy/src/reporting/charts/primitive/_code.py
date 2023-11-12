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

import json
import logging
import types
import typing as tp
from functools import cached_property
from uuid import uuid4

import autopep8
import black
import jsbeautifier

from reporting.api.types import FigureBase
from reporting.interactive.notebook_mode_inits import init_notebook_mode_for_code

logger = logging.getLogger(__name__)

_TCallableFormatter = tp.Callable[[str], str]
TFormatter = tp.Union[None, str, _TCallableFormatter]


class CodeFormattingError(Exception):
    """A class for code formatting errors"""


def _format_using_black(code: str) -> str:
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.InvalidInput as exc:
        raise CodeFormattingError from exc


def _format_using_auto_pep8(code: str) -> str:
    return autopep8.fix_code(code)


def _apply_no_formatting(code: str) -> str:
    return code


def _format_json(code: str, indent: int = 2) -> str:
    try:
        json_object = json.loads(code)
    except json.JSONDecodeError as exc:
        raise CodeFormattingError from exc
    return json.dumps(json_object, indent=indent)


def _format_js(code: str) -> str:
    return jsbeautifier.beautify(code)


AVAILABLE_FORMATTERS = types.MappingProxyType(
    {
        "py-autopep8": _format_using_auto_pep8,
        "py-black": _format_using_black,
        "json": _format_json,
        "js-jsbeautifier": _format_js,
        None: _apply_no_formatting,
    }
)


def _validate_formatter(code_formatter: TFormatter) -> _TCallableFormatter:
    if callable(code_formatter):
        return code_formatter
    elif code_formatter not in AVAILABLE_FORMATTERS:
        raise KeyError(
            f"Unknown formatter alias passed, "
            f"please consider one of known: {AVAILABLE_FORMATTERS}",
        )
    return AVAILABLE_FORMATTERS[code_formatter]


class CodePlot(FigureBase):
    def __init__(
        self,
        code: str,
        code_formatter: TFormatter = None,
        language: tp.Optional[str] = None,
    ) -> None:
        """
        Creates code object that has html representation and is convertable to html

        Args:
            code: code to plot
            code_formatter: code formatter used to prettify code.
                Has to be either a callable that converts str to formatted str
                or one of built-in formatters:
                {'py-autopep8', 'py-black', 'json', 'js-jsbeautifier'}.
                For user defined formatters note
                that in case of unsuccessful formatting
                it must raise `CodeFormattingError` to allow catch those exceptions.
            language: this will be used in representation to highlight the code.
                Use this for setting proper code highlighting in html repr in case
                auto recognised language is incorrect.
                Most common args are {'python', 'sql', 'json', 'js', 'java', 'c++'}.
                See `https://github.com/EnlighterJS/EnlighterJS#languages`
                for the full list of available languages.
        Raises:
            KeyError in case formatting is `code_formatter` is string
            and not found in available formatters
        """
        self._code = code
        self._code_formatter = _validate_formatter(code_formatter)
        self._language = language

    @cached_property
    def formatted_code(self) -> str:
        try:
            return self._code_formatter(self._code)
        except CodeFormattingError as exc:
            logger.warning(
                f"Code formatting failed, initial code will be used.\n"
                f"Code formatter: `{self._code_formatter}`.\n"
                f"Code: `{self._code}`.\n"
                f"Error message: `{exc}`.",
            )
        return self._code

    def to_html(self) -> str:
        return f'<div class="code-block">{self._repr_html_enlighter()}</div>'

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (
            f"{class_name}(code={self.formatted_code}, " f"language={self._language})"
        )

    def _repr_html_enlighter(self, code_tag_id: tp.Union[int, str, None] = None) -> str:
        id_ = uuid4() if code_tag_id is None else str(code_tag_id)
        code = self.formatted_code
        lang_attribute = (
            f'data-enlighter-language="{self._language}"'
            if self._language is not None
            else ""
        )
        code_class = 'class="enlight_js"'
        return f"<pre><code {code_class} id={id_} {lang_attribute}>{code}</code></pre>"

    def _repr_html_(self) -> str:
        """Returns HTML representation for code block"""

        # include init by default to simplify user experience (increases notebook size)
        init_notebook_mode_for_code(run_highlighter_for_all_code_blocks=False)

        uid = str(uuid4())
        activation_script = (
            "<script>"
            "EnlighterJS.enlight("
            f"  document.getElementById('{uid}'),"
            "   {rawcodeDbclick: true, toolbarBottom: false}"
            ");"
            "</script>"
        )
        return self._repr_html_enlighter(uid) + activation_script


def plot_code(
    code: str,
    code_formatter: tp.Union[None, str, _TCallableFormatter] = None,
    language: tp.Optional[str] = None,
) -> CodePlot:
    """
     Creates code object that can be shown as html object and converted to html

     Args:
         code: code to plot
         code_formatter: code formatter used to prettify code.
             Has to be either a callable that converts str to formatted str
             or one of built-in formatters:
             {'py-autopep8', 'py-black', 'json', 'js-jsbeautifier'}.
             For user defined formatters note
             that in case of unsuccessful formatting
             it must raise `CodeFormattingError` to allow catch those exceptions.
         language: this will be used in representation to highlight the code

    Raises:
         KeyError in case formatting is `code_formatter` is string
         and not found in available formatters
    """
    return CodePlot(code=code, code_formatter=code_formatter, language=language)
