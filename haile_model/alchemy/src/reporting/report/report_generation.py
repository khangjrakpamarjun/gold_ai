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
"""Functions to generate a html report."""

from __future__ import annotations

import logging
import typing as tp
from datetime import datetime
from pathlib import Path

from jinja2 import BaseLoader, DictLoader, Environment, FileSystemLoader, Template
from plotly.offline.offline import get_plotlyjs

from .rendering import TFiguresDict, TSectionDescription, render_report
from .templates import TemplateWithCodeHighlighter

logger = logging.getLogger(__name__)
DEFAULT_ENCODING = "utf-8"
_TMetaData = tp.Union[None, tp.Dict[str, tp.Any], "ReportMetaData"]


def generate_html_report(
    figures: TFiguresDict,
    render_path: tp.Union[str, Path],
    sections_description: tp.Optional[TSectionDescription] = None,
    report_meta_data: _TMetaData = None,
    report_template_path: tp.Optional[str] = None,
    report_template_assets_path: tp.Optional[str] = None,
    max_table_of_content_depth: tp.Optional[int] = 3,
    max_level_of_header: tp.Optional[int] = 3,
) -> None:
    """Creates html report of plots and dumps it to the ``output_dir``.

    Args:
        figures: **ordered** figures to include in the html report
        render_path: path to save the rendered html report
        report_meta_data: metadata of the report, used for providing additional data
            about the report; see ``ReportMetaData`` for format details
        sections_description: maps section into its description;
            key is a path to the section, value is a description.
            See ``reporting.report.rendering.render_report`` for details.
        report_template_path: path to the report template file
            If None passed, then default template is loaded
        report_template_assets_path: path to report template's assets if any are used
        max_table_of_content_depth: max header level that to show in the table of
            content; all shown if ``None`;` enumeration starts from 1
        max_level_of_header: all headers after this level will be hidden;
            none hidden if ``None``; enumeration starts from 1
    """
    render_path = Path(render_path)
    render_path.parent.mkdir(parents=True, exist_ok=True)

    rendering = render_report(
        figures,
        sections_description,
        max_table_of_content_depth,
        max_level_of_header,
    )

    jinja_template = _load_jinja_template(
        report_template_path,
        report_template_assets_path,
    )

    rendered_report = jinja_template.render(
        render=rendering.rendering_content,
        toc=rendering.table_of_content,
        meta=ReportMetaData.from_input(report_meta_data),
        plotly_js=get_plotlyjs(),  # todo: move to other resources in template
    )

    report_output_path = render_path.resolve().absolute()
    logging.info(f"Writing html report to {report_output_path}")
    render_path.write_text(rendered_report,DEFAULT_ENCODING) # TODO: previously, DEFAULT_ENCODING was missing.


def _load_jinja_template(
    template_path: tp.Optional[str],
    assets_path: tp.Optional[str],
) -> Template:
    """
    Loads provided template (with its assets) if provided;
    loads default otherwise
    """
    if template_path is None:
        template = TemplateWithCodeHighlighter()
        render_env = _create_env(loader=DictLoader(template.assets))
        template_source = template.source
    else:
        if assets_path is None:
            assets_path = Path(template_path).parent
        render_env = _create_env(loader=FileSystemLoader(assets_path))
        template_source = Path(template_path).read_text(DEFAULT_ENCODING)
    return render_env.from_string(template_source)


def _create_env(loader: BaseLoader) -> Environment:
    # We don't escape this file since it's internal file created
    # and managed by the OAI team. The file was checked and is marked as safe.
    # Adding escaping breaks file rendering.
    return Environment(loader=loader)  # noqa: S701


class ReportMetaData(object):
    DATE_TIME_FORMAT = "%Y-%m-%d %H:%M"  # noqa: WPS323

    def __init__(
        self,
        title: str = "Report Title",
        creation_time: tp.Optional[tp.Union[str, datetime]] = None,
        **kwargs: tp.Any,
    ) -> None:
        """
        Contains report metadata info

        Args:
            title: shown at the beginning of the report
            creation_time: time of report creation
            kwargs: additional metadata parameters
        """
        self.title = title
        self.creation_time = self._parse_time(creation_time)
        self._kwargs = kwargs

    @staticmethod
    def from_input(metadata: _TMetaData) -> ReportMetaData:
        if metadata is None:
            return ReportMetaData()
        if isinstance(metadata, ReportMetaData):
            return metadata
        return ReportMetaData(**metadata)

    def __getattr__(self, attribute: tp.Any) -> str:
        if isinstance(attribute, str) and attribute in self._kwargs:
            return self._kwargs[attribute]
        raise AttributeError(f"Not found attribute {attribute}")

    def _parse_time(
        self,
        time_of_creation: tp.Optional[tp.Union[str, datetime]],
    ) -> datetime:
        if time_of_creation is None:
            time_of_creation = datetime.now().strftime(self.DATE_TIME_FORMAT)
        elif isinstance(time_of_creation, str):
            try:
                time_of_creation = datetime.strptime(
                    time_of_creation,
                    self.DATE_TIME_FORMAT,
                )
            except ValueError:
                logger.warning(
                    f"Conversion to datetime format "
                    f"was unsuccessful {time_of_creation = }. Using raw string.",
                )
        return time_of_creation
