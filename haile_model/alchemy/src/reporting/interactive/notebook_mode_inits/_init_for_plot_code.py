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

from IPython.core.display import HTML, Javascript
from IPython.display import display

from reporting.report.templates import TemplateWithCodeHighlighter


def init_notebook_mode_for_code(
    run_highlighter_for_all_code_blocks: bool = True,
) -> None:
    """
    Loads the js library and corresponding styles for highlighting code

    Warning: make sure you keep the output of this cell,
    otherwise plots won't work in the expected manner.

    Args:
        run_highlighter_for_all_code_blocks: runs highlighter on all code blocks if true
    """
    _display_highlighter_styles_and_code()
    if run_highlighter_for_all_code_blocks:
        _highlight_all_code_plots_in_html()


def _highlight_all_code_plots_in_html() -> None:
    highlight_all_code = """
    for (let codeBlock of document.getElementsByClassName('enlight_js')) {
        EnlighterJS.enlight(codeBlock, {
            rawcodeDbclick: true,
            toolbarBottom: false,
        });
    }
    """
    display(Javascript(highlight_all_code))


def _display_highlighter_styles_and_code() -> None:
    template = TemplateWithCodeHighlighter()
    styles = f"<style>{template.css}</style><style>{template.css_custom}</style>"
    script = f"<script>{template.js}</script>"
    display(HTML(styles + script))
