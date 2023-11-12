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

import base64
import io
import re
import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import matplotlib.pyplot as plt
import plotly.graph_objects as go

_CSS_SECTION = "section"
_CSS_SECTION_CONTENT = "section-content"
_CSS_SECTION_TITLE = "section-title"
_CSS_FIGURE_IMG = "figure-img"
_CSS_FIGURE_HTML = "figure-html"

_INITIAL_LEVEL_HEADER = 1

_DEFAULT_IMAGE_PADDING_INCH = 0.7

_TNumericPrefix = tp.Tuple[int, ...]


@tp.runtime_checkable
class ImageCompatible(tp.Protocol):
    def savefig(
        self,
        fname: tp.BinaryIO,
        *args: tp.Any,
        format: tp.Optional[str] = None,  # noqa: WPS125
        **kwargs: tp.Any,
    ) -> None:
        """
        Saves the current figure to ``fname``;
        we pass a buffer to ``fname`` and encode the image in base64 to then show it in
        the report.

        Args:
            fname: binary file-like object to save fig to
            format: The file format, must support 'png'
        """


@tp.runtime_checkable
class HtmlCompatible(tp.Protocol):
    def to_html(self, *args: tp.Optional[tp.Any], **kwargs: tp.Optional[tp.Any]) -> str:
        """Produces html representation for report rendering purposes"""


RENDERED_TYPES = (ImageCompatible, HtmlCompatible)
TFigure = tp.Union[RENDERED_TYPES]  # type: ignore  # DEBUG - PyLance false positive
_TFigureDictKey = tp.Union[str, int]
_TFigureDictValue = tp.Union[_TFigureDictKey, TFigure, tp.List[TFigure]]
TFiguresDict = tp.Dict[_TFigureDictKey, tp.Union[_TFigureDictValue, "TFiguresDict"]]

_TFigReference = tp.Tuple[_TFigureDictKey, ...]
TSectionDescription = tp.Dict[_TFigReference, tp.Union[str, "TSectionDescription"]]


class RenderedObject(ABC):
    @abstractmethod
    def is_visible(self, visibility_level: tp.Optional[int]) -> bool:
        """Is visible method"""

    @abstractmethod
    def to_html(
        self,
        previous_item: tp.Optional[RenderedObject] = None,
        next_item: tp.Optional[RenderedObject] = None,
    ) -> str:
        """to_html method"""


class RenderedFigure(RenderedObject, ABC):
    def __init__(self, figure: TFigure) -> None:
        self._figure = figure

    @property
    def figure(self):
        return self._figure

    def __eq__(self, other: tp.Any) -> bool:
        """Helps to compare rendered objects; especially during tests"""
        if isinstance(other, RenderedFigure):
            return self._figure == other.figure
        return False

    def is_visible(self, visibility_level: tp.Optional[int]) -> bool:
        return True

    def to_html(
        self,
        previous_item: tp.Optional[RenderedObject] = None,
        next_item: tp.Optional[RenderedObject] = None,
    ) -> str:
        encoded_fig = self._encode_to_str(self._figure)
        figure_html = self._wrap_src_to_html(encoded_fig)
        prefix = (
            ""
            if isinstance(previous_item, RenderedFigure)
            else f'<div class="{_CSS_SECTION_CONTENT}">'
        )
        suffix = "" if isinstance(next_item, RenderedFigure) else "</div></div>"
        return f"{prefix}{figure_html}{suffix}"

    @staticmethod
    @abstractmethod
    def _encode_to_str(figure: TFigure) -> str:
        """encode_to_str method"""

    @staticmethod
    @abstractmethod
    def _wrap_src_to_html(encoded_src: str) -> str:
        """wrap_src_to_html method"""


class RenderedHtmlCompatible(RenderedFigure):
    @staticmethod
    def _encode_to_str(figure: HtmlCompatible) -> str:
        if isinstance(figure, go.Figure):
            html = figure.to_html(include_plotlyjs=False, full_html=False)
            return re.sub("<div>(.*)</div>", r"\1", html)  # removing wrapping div
        return figure.to_html()

    @staticmethod
    def _wrap_src_to_html(encoded_src: str) -> str:
        return f'<div class="{_CSS_FIGURE_HTML}">{encoded_src}</div>'


class RenderedImageCompatible(RenderedFigure):
    @staticmethod
    def _encode_to_str(figure: ImageCompatible) -> str:
        buf = io.BytesIO()
        if isinstance(figure, plt.Figure):
            figure.savefig(
                buf,
                format="png",
                pad_inches=_DEFAULT_IMAGE_PADDING_INCH,
                bbox_inches="tight",
            )
        else:
            figure.savefig(buf, format="png")
        fig_bytes = buf.getvalue()
        return base64.b64encode(fig_bytes).decode()

    @staticmethod
    def _wrap_src_to_html(encoded_src: str) -> str:
        return (
            f'<img class="{_CSS_FIGURE_IMG}" alt="graph" '
            f'src="data:image/png;base64,{encoded_src}">'
        )


class RenderedHeader(RenderedObject):
    # highest header level starts from 2 since h1 is reserved for report's title
    _INITIAL_H_TAG_LEVEL = 2

    def __init__(
        self,
        level: int,
        text: _TFigureDictKey,
        unique_prefix: _TNumericPrefix,
        description: tp.Optional[str],
    ) -> None:
        self._level = level
        self._text = str(text)
        self._description = description
        self._unique_prefix = "-".join(
            # Ok ignoring WPS111 for simple list/tuple/dict/set comprehension
            str(x)
            for x in unique_prefix  # noqa: WPS111
        )

    def __eq__(self, other: tp.Any) -> bool:
        """Helps to compare rendered objects; especially during tests"""
        if isinstance(other, RenderedHeader):
            return self.id == other.id
        return False

    def to_html(
        self,
        previous_item: tp.Optional[RenderedObject] = None,
        next_item: tp.Optional[RenderedObject] = None,
    ) -> str:
        h_tag = f"h{self._h_tag_level}"
        default = (
            f'<{h_tag} class="{_CSS_SECTION_TITLE}" id="header-{self.id}">'
            f"{self.text}"
            f"</{h_tag}>"
        )
        prefix = (
            ""
            if isinstance(next_item, RenderedHeader)
            else f'<div class="{_CSS_SECTION}">'
        )
        if self.description is None:
            return f"{prefix}{default}"

        info_section = (
            f'\n<span data-text="{self.description}" class="tooltip">'
            f'<span class="info-icon"></span></span>'
        )
        return f'{prefix}\n<div class="inline">{default}{info_section}</div>'

    @property
    def level(self) -> int:
        return self._level

    @property
    def text(self) -> str:
        return self._text

    @property
    def description(self) -> tp.Optional[str]:
        return self._description

    @cached_property
    def id(self) -> str:
        normalized_text = re.sub("[^0-9a-zA-Z]+", "-", self.text)
        return f"{self._unique_prefix}-{self.level}-{normalized_text}"

    def is_visible(self, visibility_level: tp.Optional[int]) -> bool:
        return self._is_visible(self.level, visibility_level)

    @property
    def _h_tag_level(self) -> int:
        """
        Evaluates level for <h> as diff between initial header level
        and initial <h> tag level
        """
        level_diff = self._INITIAL_H_TAG_LEVEL - _INITIAL_LEVEL_HEADER
        return self.level + level_diff

    @staticmethod
    def _is_visible(object_level: int, visibility_level: tp.Optional[int]) -> bool:
        if visibility_level is None:
            return True
        return object_level <= visibility_level


class RenderedTocElement(RenderedObject):
    def __init__(
        self,
        reference_header: RenderedHeader,
        children: tp.Iterable[RenderedTocElement] = (),
    ) -> None:
        self._reference_header = reference_header
        self._children: tp.List[RenderedTocElement] = list(children)

    def __eq__(self, other: tp.Any) -> bool:
        if not isinstance(other, RenderedTocElement):
            return False

        if self.reference_header_id != other.reference_header_id:
            return False

        if self.level != other.level:
            return False

        return self.children == other.children

    @property
    def text(self) -> str:
        return self._reference_header.text

    @property
    def reference_header(self) -> RenderedHeader:
        return self._reference_header

    @property
    def reference_header_id(self) -> str:
        return self._reference_header.id

    @property
    def level(self) -> int:
        return self._reference_header.level

    def is_visible(self, visibility_level: tp.Optional[int]) -> bool:
        if visibility_level is None:
            return True
        # fmt: off
        return (
            self.level <= visibility_level
            and self._reference_header.is_visible(visibility_level)
        )
        # fmt: on

    @property
    def children(self) -> tp.List[RenderedTocElement]:
        return self._children

    def add_child(self, toc_element: RenderedTocElement) -> None:
        self._children.append(toc_element)

    # todo
    def to_html(
        self,
        previous_item: tp.Optional[RenderedObject] = None,
        next_item: tp.Optional[RenderedObject] = None,
    ) -> str:
        """TODO: To be implemented"""


@dataclass
class Rendering(object):
    rendering_content: tp.List[RenderedObject]
    table_of_content: tp.List[RenderedTocElement]


def render_report(
    figures: TFiguresDict,
    sections_description: tp.Optional[TSectionDescription],
    max_table_of_content_depth: tp.Optional[int],
    max_level_of_header: tp.Optional[int],
) -> Rendering:
    """
    Renders plot

    Args:
        figures: mapping from header to section content;
            contains one or several plots or another section.
        sections_description: maps section into its description;
            key is a path to the section, value is a description. See further example.
        max_table_of_content_depth: max header level to show in the table of content;
            level indexing starts from _INITIAL_LEVEL_HEADER
        max_level_of_header: all headers after this level will be hidden;
            level indexing starts from _INITIAL_LEVEL_HEADER

    Example:
        Consider the report structure defined by following report data::
            report_data = {
                'Model Report': {'Validation Period': {'Train': ..., 'Test': ...,}}
            }

        So this example contains a section 'Model Report' on the first level
        with 'Validation Period' as its subsection.
        And on the final level we have two sections 'Train' and 'Test'.
        Let's assume we want to provide a description for each section.
        We can do so using the following structure::

            section_descriptions = {
                ('Model Report', ): '...',
                ('Model Report', 'Validation'): '...',
                ('Model Report', 'Validation', 'Train'): '...',
                ('Model Report', 'Validation', 'Test'): '...',
            }

    Returns: rendering that contains rendered objects and first level of TOC elements
    """

    rendered_section = _render_section(
        section_data=figures,
        section_descriptions=sections_description or {},
        header_level=_INITIAL_LEVEL_HEADER,
        section_prefix=(),
        section_numeric_prefix=(),
    )
    rendered_section = _hide_headers_based_on_visibility_level(
        rendered_section,
        max_level_of_header,
    )
    table_of_content = _extract_table_of_content(
        rendered_section,
        max_table_of_content_depth,
    )
    return Rendering(rendered_section, table_of_content)


def _render_section(
    section_data: TFiguresDict,
    section_descriptions: TSectionDescription,
    header_level: int,
    section_prefix: _TFigReference,
    section_numeric_prefix: _TNumericPrefix,
) -> tp.List[RenderedObject]:
    """
    Renders section:
        * creates ``RenderedHeader`` from keys
        * if content is a nested section, calls ``_render_section`` recursively,
            creates (list of) `RenderedFigure` otherwise
    """
    rendered_objects = []
    for header_index, (header, section_content) in enumerate(section_data.items()):
        current_prefix = (*section_prefix, header)
        current_numeric_prefix = (*section_numeric_prefix, header_index)
        rendered_objects.append(
            RenderedHeader(
                level=header_level,
                text=header,
                unique_prefix=current_numeric_prefix,
                description=section_descriptions.get(current_prefix),
            ),
        )
        if isinstance(section_content, dict):
            rendered_objects.extend(
                _render_section(
                    section_data=section_content,
                    section_descriptions=section_descriptions,
                    header_level=header_level + 1,
                    section_prefix=current_prefix,
                    section_numeric_prefix=current_numeric_prefix,
                ),
            )
        elif isinstance(section_content, list):
            rendered_objects.extend((_render_figure(fig) for fig in section_content))
        else:
            rendered_objects.append(_render_figure(section_content))
    return rendered_objects


def _render_figure(fig: TFigure) -> RenderedFigure:
    """
    Renders figure based on its type:
        * plotly.Figure -> RenderedPlotly
        * matplotlib.Figure -> RenderedImageCompatible
    """
    if isinstance(fig, ImageCompatible):
        return RenderedImageCompatible(fig)
    elif isinstance(fig, HtmlCompatible):
        return RenderedHtmlCompatible(fig)
    fig_type = type(fig)
    raise NotImplementedError(
        f"Current implementation supports only "
        f"``ImageCompatible`` and ``HtmlCompatible`` protocols. "
        f"Passed figure type `{fig_type}` does not support any of those.",
    )


def _hide_headers_based_on_visibility_level(
    rendered_objects: tp.List[RenderedObject],
    max_level_of_header: tp.Optional[int],
) -> tp.List[RenderedObject]:
    return [
        rendered_obj
        for rendered_obj in rendered_objects
        if rendered_obj.is_visible(max_level_of_header)
    ]


def _extract_table_of_content(
    rendered_objects: tp.List[RenderedObject],
    max_table_of_content_depth: tp.Optional[int],
) -> tp.List[RenderedTocElement]:
    """
    Selects only headers that are visible at given level of `max_table_of_content_depth`
    """
    flat_rendered_toc = [
        RenderedTocElement(rendered_object)
        for rendered_object in rendered_objects
        if isinstance(rendered_object, RenderedHeader)
    ]

    visible_toc_elements = [
        toc_element
        for toc_element in flat_rendered_toc
        if toc_element.is_visible(max_table_of_content_depth)
    ]

    if not visible_toc_elements:
        return []

    root = RenderedTocElement(RenderedHeader(-1, "", (), None))
    root.add_child(visible_toc_elements.pop(0))
    visit_stack = [root]
    for toc_element in visible_toc_elements:
        parent = visit_stack[-1]
        sibling = parent.children[-1]
        if toc_element.level > sibling.level:  # stepping down
            visit_stack.append(sibling)
            parent = sibling
        # if toc_element.level == sibling.level: then just adding to the same level
        elif toc_element.level < sibling.level:
            while toc_element.level < sibling.level:  # stepping up until found sibling
                parent = visit_stack.pop()
                sibling = parent.children[-1]
            # return parent after finding right insertion place
            visit_stack.append(parent)
        parent.add_child(toc_element)
    return root.children


def prune_to_level(
    toc: tp.List[RenderedTocElement],
    max_level: int,
) -> tp.List[RenderedTocElement]:
    if max_level is None:
        return toc
    return [
        RenderedTocElement(
            reference_header=toc_element.reference_header,
            children=prune_to_level(toc_element.children, max_level),
        )
        for toc_element in toc
        if toc_element.level <= max_level
    ]
