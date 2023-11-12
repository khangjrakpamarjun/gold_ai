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
"""Redis Dataset."""

import os
import pickle
from copy import deepcopy
from typing import Any, Dict

import redis
from kedro.io.core import AbstractDataSet, DataSetError


class RedisDataSet(AbstractDataSet):
    """`RedisDataSet` loads/saves data from/to a Redis database.

    obtained from https://github.com/kedro-org/kedro/issues/966

    Example:
        from core_pipelines.kedro_utils import RedisDataSet
        import pandas as pd

        data = pd.DataFrame({'col1': [1, 2], 'col2': [4, 5],
                                'col3': [5, 6]})

        my_data = RedisDataSet(key="my_data")
        my_data.save(data)
        reloaded = my_data.load()
        assert data.equals(reloaded)
    """

    default_redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
    DEFAULT_FROM_URL_ARGS = {"url": default_redis_url}  # type: Dict[str, Any]
    DEFAULT_SAVE_ARGS = {}  # type: Dict[str, Any]

    def __init__(
        self,
        key: str,
        from_url_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
    ) -> None:
        """Construct instance of `RedisDataSet`.

        Args:
            key: The key to use for saving/ loading object to Redis redis_url:
            Redis database url. Otherwise it is extracted from environment
            variables.
            from_url_args: Additional arguments to pass to `redis.Redis.from_url`.
            save_args: Additional arguments to pass to `redis.Redis.set`.

        """
        super().__init__()
        self._key = key
        self._from_url_args = deepcopy(self.DEFAULT_FROM_URL_ARGS)
        if from_url_args is not None:
            self._from_url_args.update(from_url_args)
        self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)
        self._redis_db = redis.Redis.from_url(**self._from_url_args)

    def _describe(self) -> Dict[str, Any]:
        return dict(key=self._key, **self._from_url_args)

    # `redis_db` mypy does not work since it is optional and optional is not
    # accepted by pickle.loads.
    def _load(self) -> Any:
        return pickle.loads(self._redis_db.get(self._key))  # type: ignore

    def _save(self, data: Any) -> None:
        try:
            self._redis_db.set(self._key, pickle.dumps(data), **self._save_args)
        except Exception as exc:
            raise DataSetError(
                f"{data.__class__} was not serialized due to: {exc}",
            ) from exc

    def _exists(self) -> bool:
        try:
            key_exists = bool(self._redis_db.exists(self._key))
        except DataSetError:
            return False

        return key_exists
