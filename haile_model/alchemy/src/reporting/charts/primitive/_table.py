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

import logging
import typing as tp
from itertools import chain

import numpy as np
import pandas as pd
from itables import JavascriptCode, JavascriptFunction
from itables import to_html_datatable as itables_to_html_datatable
from pandas.api.types import is_numeric_dtype

from reporting.api.types import FigureBase
from reporting.charts.utils import check_data

DEFAULT_TITLE_STYLE = (
    "caption-side: top; "
    "font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; "
    "font-size: 17px; "
    "color: rgb(42, 63, 95); "
    "opacity: 1; "
    "font-weight: normal; "
    "white-space: pre; "
)

_TDict = tp.Dict[str, tp.Any]
_TColumnName = tp.Union[str, int]
_TSortingConfig = tp.Optional[tp.List[tp.Tuple[_TColumnName, str]]]
_TByColumnPrecision = tp.Dict[tp.Hashable, int]
_TPrecision = tp.Optional[tp.Union[int, _TByColumnPrecision]]

_DEFAULT_PRECISION_KEY = "_default"

_VALID_SORT_ARGS = frozenset(("asc", "desc"))

logger = logging.getLogger(__name__)


def _select_columns(data: pd.DataFrame, columns: tp.List[str]) -> pd.DataFrame:
    if columns:
        check_data(data, *columns)
        data = data[columns]
    else:
        check_data(data, validate_columns=False)
    return data


class TablePlot(FigureBase):
    def __init__(
        self,
        data: tp.Union[pd.Series, pd.DataFrame],
        columns: tp.Optional[tp.List[str]] = None,
        precision: _TPrecision = 4,
        title: tp.Optional[str] = None,
        columns_filters_position: tp.Optional[str] = "footer",
        columns_to_color_as_bars: tp.Optional[tp.List[str]] = None,
        width: float = 50,
        table_alignment: str = "left",
        sort_by: _TSortingConfig = None,
        show_index: bool = True,
    ) -> None:
        """
        Table chart implemented using `itables`.
        This function provides a fancier (compared to plotly) representation of tables
        (with column ordering and filters).

        Args:
            data: dataframe for chart
            columns: column names of column data to show in the table chart
            precision: sets the number of digits to round float values; can be either:
                * int - sets same precision for every numeric column
                * dict - mapping from column name to its precision;
                  "_default" key is optional and confiures the rest of the columns
            title: title for the chart
            columns_filters_position: position for placing columns filter,
                one of {None, "left", "right"}; hidden when `None` is passed
            columns_to_color_as_bars: list of column names that will be barcolored
                using the value inside it
            width: the width of the table in layout in percentage
            table_alignment: table alignment, can be one of {"none", "left", "right"}
            sort_by: list for default columns sorting;
                each list element represents a column name and
                its order either "asc" or "desc".
                Keep index named to be able to sort it, if there is a column
                and index with the same name present and `show_index` is set True,
                then column name will be used for sorting and warning will be thrown
            show_index: shows index column if set True

        Examples::
            Sort by `column_one` ascending and by `column_two` descending
            >>> df = pd.DataFrame({"column_one": [1, 3, 2], "column_two": [6, 4, 5]})
            >>> plot_table(df, sort_by=[("column_one", "asc"), ("column_two", "desc")])

            Sort by `column_one` ascending and by `column_two` descending
            >>> df = pd.DataFrame({"index": [1, 3, 2], "column": [6, 4, 5]}).set_index("index")
            >>> plot_table(df, sort_by=[("index", "asc"), ("column_two", "desc")])

            Set precision for all columns
            >>> df = pd.DataFrame({"column_one": [1, 3, 2], "column_two": [6, 4, 5]})
            >>> plot_table(df, precision=2)

            Set precision for specific column and default precision for rest
            >>> df = pd.DataFrame({"column_one": [1, 3, 2], "column_two": [6, 4, 5]})
            >>> plot_table(df, precision={"column_one": 1, "_default": 0})

        Returns: plotly table chart
        """  # noqa: E501  # Ok to render the line of code nicely
        self.data = data.to_frame() if isinstance(data, pd.Series) else data
        self.columns = columns
        self.precision = precision
        self.title = title
        self.width = width
        self.table_alignment = table_alignment
        self.columns_filters_position = columns_filters_position
        self.columns_to_color_as_bars = _validate_bars_config(
            self.data,
            columns_to_color_as_bars,
            show_index,
        )
        self.sort_by = _validate_sorting_config(self.data, sort_by, show_index)
        self.show_index = show_index

    def to_html(self) -> str:
        return self._repr_html_()

    def _repr_html_(self) -> str:
        prepared_data = _select_columns(self.data, self.columns)
        pre_table_tags = (
            (
                f'<caption class="gtitle" style="{DEFAULT_TITLE_STYLE}">'
                f"{self.title}</caption>"
            )
            if self.title is not None
            else ""
        )
        columns_filters_position = (
            self.columns_filters_position
            if self.columns_filters_position is not None
            else False
        )
        return itables_to_html_datatable(
            prepared_data,
            style=f"table-layout:auto;width:{self.width}%;float:{self.table_alignment}",
            classes="hover order-column",
            tags=pre_table_tags,
            column_filters=columns_filters_position,
            order=self.sort_by,
            showIndex=self.show_index,
            columnDefs=[
                {"className": "dt-body-right dt-head-left", "targets": "_all"},
                *_get_formatting_spec(
                    prepared_data.dtypes.to_dict(),
                    self.show_index,
                    self.precision,
                ),
                *_get_bar_coloring_spec(
                    self.data,
                    self.columns_to_color_as_bars,
                    self.show_index,
                ),
            ],
        )


def _validate_sorting_config(
    data: pd.DataFrame,
    sort_by: _TSortingConfig,
    show_index: bool,
) -> tp.List[tp.Tuple[int, str]]:
    """
    * Maps all column names to itables format i.e. list of (column index, column order)
    * Validates that all columns are present,
      no duplicated columns are passed, all sorting args are either `asc` or `desc`.

    Notes:
        If there is a column and index with the same name present,
        column name will be used for sorting and warning thrown
    """
    if sort_by is None:
        return []

    if not isinstance(sort_by, list):
        received_type = type(sort_by)
        raise ValueError(
            f"`sort_by` must be of type `None | list`; received: {received_type}",
        )

    index_and_column_names = _get_index_and_column_names(data, show_index)
    sort_by_columns = [column for column, _ in sort_by]
    _validate_all_columns_are_known(index_and_column_names, sort_by_columns, "sort_by")
    _validate_no_duplicated_columns(sort_by_columns, raise_error=True)
    _validate_sort_by_order_argument(sort_by)

    index_and_columns_intersected_with_sort = [
        column for column in index_and_column_names if column in set(sort_by_columns)
    ]  # we keep counts for each column to then search for duplicated
    _validate_no_duplicated_columns(
        index_and_columns_intersected_with_sort,
        raise_error=False,
    )

    column_to_its_index = {
        column: column_index
        for column_index, column in enumerate(index_and_column_names)
    }
    return [(column_to_its_index[column], order) for column, order in sort_by]


def _get_index_and_column_names(data: pd.DataFrame, show_index: bool):
    """Gets a list of all the names, starting with the index name and then listing all
    the column names.
    """
    indexes_names = data.index.names if show_index else []
    return list(chain(indexes_names, data.columns))


def _validate_bars_config(
    data: pd.DataFrame,
    columns_to_color_as_bars: tp.Optional[tp.List[str]],
    show_index: bool,
) -> tp.List[tp.Tuple[int, str]]:
    """
    Validates that:
        * all columns are present
        * no duplicated columns are passed and
        * requested columns are numeric

    Notes:
        If there is a column and index with the same name present,
        column name will be used for bar coloring and warning thrown
    """
    if columns_to_color_as_bars is None:
        return []

    if not isinstance(columns_to_color_as_bars, list):
        received_type = type(columns_to_color_as_bars)
        raise ValueError(
            f"`columns_to_color_as_bars` must be of type `None | list`; "
            f"received: {received_type}",
        )

    indexes_names = data.index.names if show_index else []
    index_and_column_names = list(chain(indexes_names, data.columns))

    arg_name = "columns_to_color_as_bars"
    _validate_all_columns_are_known(
        index_and_column_names,
        columns_to_color_as_bars,
        arg_name,
    )

    index_and_columns_intersected_with_arg = [
        column
        for column in index_and_column_names
        if column in set(columns_to_color_as_bars)
    ]  # we keep counts for each column to then search for duplicated
    _validate_no_duplicated_columns(
        index_and_columns_intersected_with_arg,
        raise_error=False,
    )
    _validate_requested_columns_are_numeric(
        data.reset_index().dtypes.to_dict(),  # add index to dtypes
        index_and_columns_intersected_with_arg,
        arg_name,
    )

    column_to_its_index = {
        column: column_index
        for column_index, column in enumerate(index_and_column_names)
    }
    return [
        (column_to_its_index[column], column) for column in columns_to_color_as_bars
    ]


def _validate_all_columns_are_known(
    index_and_column_names: tp.List[str],
    sort_by_columns: tp.List[str],
    requesting_arg_name: str,
) -> None:
    unknown_columns = set(sort_by_columns).difference(index_and_column_names)
    if unknown_columns:
        raise ValueError(
            f"Found unknown columns requested in "
            f"`{requesting_arg_name}`: {unknown_columns}",
        )


def _validate_no_duplicated_columns(
    sort_by_columns: tp.List[str],
    raise_error: bool,
) -> None:
    unique_columns, counts = np.unique(sort_by_columns, return_counts=True)
    duplicated_columns = list(unique_columns[counts > 1])
    if duplicated_columns:
        err_message = f"Found duplicated columns: {duplicated_columns}"
        if raise_error:
            raise ValueError(err_message)
        else:
            logger.warning(err_message)


def _validate_requested_columns_are_numeric(
    dtypes: tp.Mapping[str, np.dtype],
    columns: tp.List[str],
    requesting_arg_name: str,
) -> None:
    non_numeric_columns = [
        column for column in columns if not is_numeric_dtype(dtypes[column])
    ]
    if non_numeric_columns:
        raise ValueError(
            f"Found non-numeric columns requested by {requesting_arg_name}: "
            f"{non_numeric_columns}",
        )


def _validate_sort_by_order_argument(sort_by: _TSortingConfig) -> None:
    invalid_sorting_args = [
        (columns, sort_arg)
        for columns, sort_arg in sort_by
        if sort_arg not in _VALID_SORT_ARGS
    ]
    if invalid_sorting_args:
        raise ValueError(f"Invalid sorting arg(s) found: {invalid_sorting_args}")


def _by_column_precision_from_int(
    data_dtypes: tp.Mapping[str, np.dtype],
    precision: int,
):
    """By-column precision for all numeric types in ``data_dtypes`` assigned as the
    provided ``precision`` value

    Returns a dict where the entries are the names of the numeric columns, and the
    values for all of them is the desired precision ``precision``
    """
    return {
        column: precision
        for column, dtype in data_dtypes.items()
        if is_numeric_dtype(dtype)
    }


def _by_columns_precision_from_dict(
    data_dtypes: tp.Mapping[str, np.dtype],
    precision: tp.Dict[tp.Hashable, int],
):
    precision = precision.copy()
    default_precision = precision.pop(_DEFAULT_PRECISION_KEY, None)
    extra_columns = set(precision).difference(data_dtypes)
    if extra_columns:
        raise ValueError(
            f"For provided precision spec, "
            f"couldn't find following columns in the data: {extra_columns}",
        )
    by_column_precisions = {
        column: precision
        for column, precision in precision.items()
        if column in data_dtypes
    }
    if default_precision is not None:
        by_column_precisions.update(
            {
                column: default_precision
                for column, dtype in data_dtypes.items()
                if column not in precision and is_numeric_dtype(dtype)
            },
        )
    return by_column_precisions


def _get_formatting_spec(
    data_dtypes: tp.Mapping[str, np.dtype],
    show_index: bool,
    precision: _TPrecision,
) -> tp.List[tp.Dict[str, str]]:
    """
    Parses input precision spec

    Args:
        data_dtypes: dataframe's dtypes in dict format
        show_index: true if index is shown
        precision: precision spec

    Returns:
        Parsed itables config
    """
    if precision is None:
        return []

    if isinstance(precision, int):
        by_column_precisions = _by_column_precision_from_int(data_dtypes, precision)
    elif isinstance(precision, dict):
        by_column_precisions = _by_columns_precision_from_dict(data_dtypes, precision)
    else:
        raise ValueError("Precision must be either `int` or `dict[Hashable, int]`")
    _validate_precisions_are_greater_or_equal_than_zero(by_column_precisions)
    precision_spec = _get_precision_spec_for_data_columns(
        data_dtypes,
        by_column_precisions,
        show_index=show_index,
    )
    return precision_spec  # noqa: WPS331  # Naming makes meaning clearer


def _get_precision_spec_for_data_columns(
    data_dtypes: tp.Mapping[str, np.dtype],
    by_column_precisions,
    show_index: bool,
):
    """Get the precision specifications for the columns specified in
    ``by_column_precision``, using their index in the ``data_columns``
    """
    starting_index = 1 if show_index else 0
    data_columns = list(data_dtypes)
    precision_spec = [
        _get_formatting_spec_for_column(
            column_precision,
            data_columns.index(column) + starting_index,
        )
        for column, column_precision in by_column_precisions.items()
    ]
    return precision_spec  # noqa: WPS331  # Naming makes meaning clearer


# WPS118 in the line below is ok if naming makes meaning clearer
def _validate_precisions_are_greater_or_equal_than_zero(  # noqa: WPS118
    by_column_precisions: tp.Dict[str, int],
) -> None:
    wrong_precisions = [
        column
        for column, column_precision in by_column_precisions.items()
        if column_precision < 0
    ]
    if wrong_precisions:
        raise ValueError(f"Found `precision` < 0 for columns: {wrong_precisions}")


def _get_formatting_spec_for_column(
    precision: int,
    column: int,
    thousands_sep: str = ",",
    precision_delimiter: str = ".",
) -> tp.Dict[str, str]:
    """
    Returns formatting spec {
        "targets": targets_specification,
        "render": js_rendering_code,
    }

    Args:
        precision: number of digits in float representation
        column: column index
    """
    js_formatting_function = (
        "$.fn.dataTable.render.number("
        "'{thousands_sep}', '{precision_delimiter}', {precision})"
    ).format(
        thousands_sep=thousands_sep,
        precision_delimiter=precision_delimiter,
        precision=precision,
    )
    return {
        "targets": column,
        "render": JavascriptCode(js_formatting_function),
    }


def _get_bar_coloring_spec(
    df: pd.DataFrame,
    columns_to_draw_bar_for: tp.List[tp.Tuple[int, str]],
    show_index: bool,
) -> tp.List[tp.Dict[str, tp.Any]]:
    if columns_to_draw_bar_for is None:
        return []

    if show_index:  # we do that to be able to get index column by name
        # to support older pandas we manually do `df.reset_index(allow_duplicates=True)`
        index_columns = df.index.names
        index_to_reset_into_columns = (
            set(index_columns).difference(df.columns) if index_columns else set()
        )
        df = df.reset_index(level=list(index_to_reset_into_columns))

    return [
        {
            "target": column_index,
            "createdCell": _get_bar_coloring_config_for_column(df[column]),
        }
        for column_index, column in columns_to_draw_bar_for
    ]


def _get_bar_coloring_config_for_column(column_data: pd.Series) -> JavascriptFunction:
    # todo: start coloring based on positive and negative values
    column_data = column_data.abs()  # calculate bar size based on abs values
    column_min = column_data.min()
    column_range = column_data.max() - column_data.min()
    javascript_coloring_fn = """
        function f(td, cellData, rowData, row, col) {
            let cellDataNormalized = (
                Math.abs(cellData) - $COLUMN_MIN
            ) / $COLUMN_RANGE * 100;
            let transparency = Math.max(100 - cellDataNormalized - 1, 0);
            $(td).css(
                'background-image',
                `linear-gradient(
                    90deg, transparent ${transparency}%, lightblue ${transparency}%
                )`
            );
            $(td).css('background-size', '98% 88%');
            $(td).css('background-position', 'center center');
            $(td).css('background-repeat', 'no-repeat no-repeat');
        }
        """
    return JavascriptFunction(
        javascript_coloring_fn.replace("$COLUMN_MIN", str(column_min)).replace(
            "$COLUMN_RANGE", str(column_range)
        ),
    )


def plot_table(
    data: pd.DataFrame,
    columns: tp.Optional[tp.List[str]] = None,
    precision: _TPrecision = 4,
    title: tp.Optional[str] = None,
    columns_filters_position: tp.Optional[str] = "footer",
    columns_to_color_as_bars: tp.Optional[tp.List[str]] = None,
    width: float = 50,
    table_alignment: str = "left",
    sort_by: _TSortingConfig = None,
    show_index: bool = True,
) -> FigureBase:
    """
    Returns table chart rendered using `itables`.
    This function provides a fancier (compared to plotly) representation of tables
    (with column ordering and filters).

    Args:
        data: dataframe for chart
        columns: column names of column data to show in the table chart
        precision: sets the number of digits to round float values
        title: title for the chart
        columns_filters_position: position for placing columns filter,
            one of {None, "header", "footer"}; hidden when `None` is passed
        columns_to_color_as_bars: list of column names that will be barcolored
            using the value inside it
        width: the width of the table in layout in percentage
        table_alignment: table alignment, can be one of {"none", "left", "right"}
        sort_by: list for default columns sorting;
            each list element represents a column name and
            its order either "asc" or "desc"
        show_index: shows index column if set True
    Examples::
        Sort by `column_one` ascending and by `column_two` descending
        >>> df = pd.DataFrame({"column_one": [1, 3, 2], "column_two": [6, 4, 5]})
        >>> plot_table(df, sort_by=[("column_one", "asc"), ("column_two", "desc")])

        Sort by `column_one` ascending and by `column_two` descending
        >>> df = pd.DataFrame({"index": [1, 3, 2], "column": [6, 4, 5]}).set_index("index")
        >>> plot_table(df, sort_by=[("index", "asc"), ("column_two", "desc")])

        Set precision for all columns
        >>> df = pd.DataFrame({"column_one": [1, 3, 2], "column_two": [6, 4, 5]})
        >>> plot_table(df, precision=2)

        Set precision for specific column and default precision for rest
        >>> df = pd.DataFrame({"column_one": [1, 3, 2], "column_two": [6, 4, 5]})
        >>> plot_table(df, precision={"column_one": 1, "_default": 0})

    Returns: plotly table chart
    """  # noqa: E501  # Ok in order to have the code in the string look nice

    return TablePlot(
        data=data,
        columns=columns,
        precision=precision,
        title=title,
        columns_filters_position=columns_filters_position,
        columns_to_color_as_bars=columns_to_color_as_bars,
        width=width,
        table_alignment=table_alignment,
        sort_by=sort_by,
        show_index=show_index,
    )
