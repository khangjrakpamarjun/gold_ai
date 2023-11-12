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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

TEMPLATES_PATH = Path(__file__).parent / "html_template"
DEFAULT_ENCODING = "utf-8"


@dataclass
class TemplateBase(ABC):
    @property
    @abstractmethod
    def source(self) -> str:
        """A ``source`` property to store the path to the template (html) file"""

    @property
    @abstractmethod
    def assets(self) -> tp.Dict[str, str]:
        """An ``assets`` property to store a dict with all the needed resources
        needed for the template to be used to generate reports (e,g, .css and .js
        files)
        """


@tp.final
class TemplateWithCodeHighlighter(TemplateBase):
    def __init__(self) -> None:
        self._template_name = "template.html"
        self._js_name = "enlighterjs.min.js"
        self._css_name = "enlighterjs.min.css"
        self._css_custom_name = "enlighterjs_custom.css"
        self._load_template()

    @property
    def source(self) -> str:
        return self._source

    @property
    def assets(self) -> tp.Dict[str, str]:
        return self._assets.copy()

    @property
    def js(self) -> str:
        return self._assets[self._js_name]

    @property
    def css(self) -> str:
        return self._assets[self._css_name]

    @property
    def css_custom(self) -> str:
        return self._assets[self._css_custom_name]

    def _load_template(self) -> None:
        """Loads html template's source code and assets"""
        self._source = (TEMPLATES_PATH / self._template_name).read_text(
            DEFAULT_ENCODING
        )
        resources_to_load = (self._js_name, self._css_name, self._css_custom_name)
        self._assets = {
            resource: (TEMPLATES_PATH / resource).read_text(DEFAULT_ENCODING)
            for resource in resources_to_load
        }
