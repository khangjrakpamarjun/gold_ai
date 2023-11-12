"""``SQLDataSet`` to load and save data to a SQL backend."""

import copy
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, NoReturn, Optional

import numpy as np
import pandas as pd
from kedro.config import ConfigLoader, TemplatedConfigLoader
from kedro.io.core import AbstractDataSet, DataSetError
from sqlalchemy import create_engine
from sqlalchemy.exc import NoSuchModuleError

logger = logging.getLogger(__name__)

__all__ = ["TimeScaleDBSqlQueryDataSet"]

KNOWN_PIP_INSTALL = {
    "psycopg2": "psycopg2",
    "mysqldb": "mysqlclient",
    "cx_Oracle": "cx_Oracle",
}

DRIVER_ERROR_MESSAGE = """
A module/driver is missing when connecting to your SQL server. SQLDataSet
 supports SQLAlchemy drivers. Please refer to
 https://docs.sqlalchemy.org/en/13/core/engines.html#supported-databases
 for more information.
\n\n
"""


def _find_known_drivers(module_import_error: ImportError) -> Optional[str]:
    """Looks up known keywords in a ``ModuleNotFoundError`` so that it can
    provide better guideline for the user.

    Args:
        module_import_error: Error raised while connecting to a SQL server.

    Returns:
        Instructions for installing missing driver. An empty string is
        returned in case error is related to an unknown driver.

    """

    # module errors contain string "No module name 'module_name'"
    # we are trying to extract module_name surrounded by quotes here
    res = re.findall(r"'(.*?)'", str(module_import_error.args[0]).lower())

    # in case module import error does not match our expected pattern
    # we have no recommendation
    if not res:
        return None

    missing_module = res[0]

    if KNOWN_PIP_INSTALL.get(missing_module):
        return (
            "You can also try installing missing driver with\n"
            f"\npip install {KNOWN_PIP_INSTALL.get(missing_module)}"
        )

    return None


def _get_missing_module_error(import_error: ImportError) -> DataSetError:
    missing_module_instruction = _find_known_drivers(import_error)

    if missing_module_instruction is None:
        return DataSetError(
            f"{DRIVER_ERROR_MESSAGE}Loading failed with error:\n\n{str(import_error)}"
        )

    return DataSetError(f"{DRIVER_ERROR_MESSAGE}{missing_module_instruction}")


def _get_sql_alchemy_missing_error() -> DataSetError:
    return DataSetError(
        "The SQL dialect in your connection is not supported by "
        "SQLAlchemy. Please refer to "
        "https://docs.sqlalchemy.org/en/13/core/engines.html#supported-databases "
        "for more information."
    )


def _get_conf():
    conf_path = (
        os.path.split(os.path.dirname(os.path.dirname(Path(__file__).parent.parent)))[0]
        + "/conf"
    )
    return TemplatedConfigLoader(conf_source=conf_path, globals_pattern="*globals.yml")


def _get_tag_dictionary_filepath() -> Path:
    return _get_conf()["catalog"]["td"]["filepath"]


def get_end_time():
    if _get_conf()["parameters"]["run_end_date"] != "":
        return _get_conf()["parameters"]["run_end_date"]
    else:
        return None


class TimeScaleDBSqlQueryDataSet(AbstractDataSet[None, pd.DataFrame]):
    engines: Dict[str, Any] = {}

    def __init__(  # pylint: disable=too-many-arguments
        self,
        load_args: Dict[str, Any] = None,
        write_table_name: str = "",
        sql: str = "",
        last_month: bool = False,
    ) -> None:
        """Creates a new ``TimeScaleDBSqlQueryDataSet`` for reading data
        from Timescale Database. After fetching the data, it will be
        filtered as per tag_dictionary required values

        Args:
            load_args: Provided to underlying pandas ``read_sql_query``
                function along with the connection string.
                To find all supported arguments, see here:
                https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_query.html
                To find all supported connection string formats, see here:
                https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls
        """

        default_load_args = {}  # type: Dict[str, Any]

        self.tag_dictionary_path = _get_tag_dictionary_filepath()
        self.write_table_name = write_table_name
        self.sql = sql
        self.last_month = last_month

        self._load_args = (
            {**default_load_args, **load_args}
            if load_args is not None
            else default_load_args
        )

        dbname = os.environ.get("PGDATABASE")
        user = os.environ.get("PGUSER")
        port = os.environ.get("PGPORT") or "5432"
        host = os.environ.get("PGHOST")
        password = os.environ.get("PGPASSWORD")
        self._connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        self.create_connection(self._connection_str)

    @classmethod
    def create_connection(cls, connection_str: str) -> None:
        """Given a connection string, create singleton connection
        to be used across all instances of `SQLQueryDataSet` that
        need to connect to the same source.
        """
        if connection_str in cls.engines:
            return

        try:
            engine = create_engine(connection_str)
        except ImportError as import_error:
            raise _get_missing_module_error(import_error) from import_error
        except NoSuchModuleError as exc:
            raise _get_sql_alchemy_missing_error() from exc

        cls.engines[connection_str] = engine

    def _describe(self) -> Dict[str, Any]:
        load_args = copy.deepcopy(self._load_args)
        return {
            "sql": str(load_args.pop("sql", None)),
            "load_args": str(load_args),
        }

    def _load(self) -> pd.DataFrame:
        # TODO: Change below step to read td directly from catalog
        df = pd.read_csv(self.tag_dictionary_path)

        list_of_tags = df[(df["data_source"].isin(["PI"]))]["tag"].to_list()
        end_time = get_end_time()
        if self.last_month:
            if end_time:
                sql_query = (
                    self.sql + " where timestamp >"
                    f" '{pd.to_datetime(end_time) - timedelta(days=30)}' and"
                    f" timestamp <= '{pd.to_datetime(end_time)}'"
                )
            else:
                sql_query = (
                    self.sql + " where timestamp >"
                    f" '{pd.to_datetime(datetime.now() - timedelta(days=30))}'"
                )
        else:
            sql_query = self.sql

        logger.info("querying data with:%s", sql_query)
        engine = self.engines[self._connection_str]
        df = pd.read_sql_query(sql_query, con=engine)
        df = df[df["tag"].isin(list_of_tags)]
        df = pd.pivot_table(
            df,
            values="value",
            index=["timestamp"],
            columns=["tag"],
            aggfunc=np.max,
        ).reset_index()

        return df

    def _save(self, df: pd.DataFrame) -> NoReturn:
        engine = self.engines[self._connection_str]

        df.to_sql(self.write_table_name, con=engine, if_exists="append", index=False)
