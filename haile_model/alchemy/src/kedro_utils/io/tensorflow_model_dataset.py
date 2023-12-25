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

import copy
import typing as tp
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import fsspec
import tensorflow as tf
from kedro.extras.datasets.pickle import PickleDataSet
from kedro.extras.datasets.tensorflow import TensorFlowModelDataset
from kedro.io import AbstractVersionedDataSet, Version
from kedro.io.core import get_filepath_str, get_protocol_and_path


class WrapsKerasModel(tp.Protocol):
    """
    A type definition of a class that has `._keras_model`
    attribute that stores `tf.keras.Model`.
    `modeling.KerasModel` is aligned with this type definition.
    """

    @property
    def _keras_model(self) -> tf.keras.Model:
        """This property stores a ``tf.Keras.Model``"""

    @_keras_model.setter
    def _keras_model(self, data: tf.keras.Model) -> None:
        """Setter method for the keras model"""


class KerasModelDataset(AbstractVersionedDataSet[WrapsKerasModel, WrapsKerasModel]):
    """``KerasModelDataset`` loads and saves KerasModel instances.
    The underlying functionality is supported by,
    and passes input arguments through to,
    TensorFlow 2.X tensorflow_dataset_load_args
    and tensorflow_dataset_save_args methods.

    Example using Python API:
    ::

        >>> from kedro_utils.io.tensorflow_model_dataset import KerasModelDataset
        >>> from modeling.load.keras_model import KerasModel
        >>> import tensorflow as tf
        >>>
        >>> data_set = KerasModelDataset("data/06_models/sample_keras_model")
        >>> model = KerasModel(...)
        >>> predictions = model.predict([...])
        >>>
        >>> data_set.save(model)
        >>> loaded_model = data_set.load()
        >>> new_predictions = loaded_model.predict([...])
        >>> np.testing.assert_allclose(
        ...     predictions,
        ...     new_predictions,
        ...     rtol=1e-6,
        ...      atol=1e-6
        ... )

    """

    _DIR_NAME_FOR_TENSORFLOW_MODEL: str = "tensorflow_model"
    _FILENAME_FOR_WRAPPER_OBJECT: str = "keras_model_wrapper.pkl"

    def __init__(
        self,
        filepath: str,
        backend: str = "pickle",
        tensorflow_dataset_load_args: tp.Optional[tp.Dict[str, tp.Any]] = None,
        pickle_dataset_load_args: tp.Optional[tp.Dict[str, tp.Any]] = None,
        tensorflow_dataset_save_args: tp.Optional[tp.Dict[str, tp.Any]] = None,
        pickle_dataset_save_args: tp.Optional[tp.Dict[str, tp.Any]] = None,
        version: Version = None,
        credentials: tp.Dict[str, tp.Any] = None,
        fs_args: tp.Dict[str, tp.Any] = None,
    ) -> None:
        """
        Creates a new instance of ``KerasModelDataset``.

        Args:
            filepath: Filepath in POSIX format to a KerasModel
                model directory prefixed with a
                protocol like `s3://`. If prefix is not
                provided `file` protocol (local filesystem)
                will be used. The prefix should be any protocol supported by ``fsspec``.
                Note: `http(s)` doesn't support versioning.
            backend: PickleDataSet `backend` initialization parameter
                for storing the model wrapper
            tensorflow_dataset_load_args: TensorFlow options for loading models.
                Used for `load_args` parameter for initializing TensorFlowModelDataset
            pickle_dataset_load_args: Pickle options for loading pickle files.
                Used for `load_args` param for initializing PickeDataset.
            tensorflow_dataset_save_args: TensorFlow options for sacing models.
                Used for `save_args` parameter for initializing TensorFlowModelDataset
            pickle_dataset_save_args: Pickle options for saving pickle files.
                Used for `save_args` param for initializing PickeDataset.
            version: If specified, should be an instance of
                ``kedro.io.core.Version``. If its ``load`` attribute is
                None, the latest version will be loaded. If its ``save``
                attribute is None, save version will be autogenerated.
            credentials: Credentials required to get access to the
                underlying filesystem.
                E.g. for ``GCSFileSystem`` it should look like `{"token": None}`.
            fs_args: Extra arguments to pass into underlying
                filesystem class constructor
                (e.g. `{"project": "my-project"}` for ``GCSFileSystem``), as well as
                to pass to the filesystem's `open` method through nested keys
                `open_args_load` and `open_args_save`.
                Here you can find all available arguments for `open`:
                https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.open
                All defaults are preserved, except `mode`,
                 which is set to `wb` when saving.
        """
        fs_args = copy.deepcopy(fs_args) or {}
        credentials = copy.deepcopy(credentials) or {}
        protocol, path = get_protocol_and_path(filepath, version)
        if protocol == "file":
            fs_args.setdefault("auto_mkdir", True)

        self._protocol = protocol
        self._fs = fsspec.filesystem(self._protocol, **credentials, **fs_args)
        self._tensorflow_dataset_load_args = tensorflow_dataset_load_args
        self._pickle_dataset_load_args = pickle_dataset_load_args
        self._tensorflow_dataset_save_args = tensorflow_dataset_save_args
        self._pickle_dataset_save_args = pickle_dataset_save_args
        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )
        self._pickle_dataset = PickleDataSet(
            str((Path(path) / self._FILENAME_FOR_WRAPPER_OBJECT).resolve()),
            backend,
            load_args=pickle_dataset_load_args,
            save_args=pickle_dataset_save_args,
            version=None,
            credentials=credentials,
            fs_args=fs_args,
        )
        self._tensorflow_dataset = TensorFlowModelDataset(
            str((Path(path) / self._DIR_NAME_FOR_TENSORFLOW_MODEL).resolve()),
            load_args=tensorflow_dataset_load_args,
            save_args=tensorflow_dataset_save_args,
            version=None,
            credentials=credentials,
            fs_args=fs_args,
        )

    @contextmanager
    def _tell_version_to_private_datasets(self) -> None:
        """
        Context manager modifying filepaths for inner kedro-based datasets
        to reflect version set from the outer dataset.

        Notes:
            If versioning is enabled for kedro-based dataset, filepath is
            being modified reflecting version.
            Inner datasets have versioning disabled and need to know about
            versioning from outer datasets.
        """
        version = self.resolve_save_version()
        tensorflow_dataset_filepath = self._tensorflow_dataset._filepath
        pickle_dataset_filepath = self._pickle_dataset._filepath
        try:  # noqa: WPS229
            if version is not None:
                self._tensorflow_dataset._filepath = (
                    self._tensorflow_dataset._filepath.parent
                    / version
                    / self._filepath.name
                    / self._tensorflow_dataset._filepath.name
                )
                self._pickle_dataset._filepath = (
                    self._pickle_dataset._filepath.parent
                    / version
                    / self._filepath.name
                    / self._pickle_dataset._filepath.name
                )
            yield
        finally:
            self._tensorflow_dataset._filepath = tensorflow_dataset_filepath
            self._pickle_dataset._filepath = pickle_dataset_filepath

    def _save(self, data: WrapsKerasModel) -> None:
        self._get_save_path()
        with self._tell_version_to_private_datasets():
            self._tensorflow_dataset._save(data._keras_model)
            keras_model = data._keras_model
            data._keras_model = None
            self._pickle_dataset._save(data)
        data._keras_model = keras_model

    def _load(self) -> WrapsKerasModel:
        self._get_load_path()
        with self._tell_version_to_private_datasets():
            keras_model = self._tensorflow_dataset._load()
            class_model = self._pickle_dataset._load()
        if hasattr(class_model, "_keras_model"):  # noqa: WPS421
            class_model._keras_model = keras_model
            return class_model
        raise RuntimeError(
            "Error loading KerasModel, specified" " file is not a KerasModel instance.",
        )

    def _exists(self) -> bool:
        self._get_load_path()
        with self._tell_version_to_private_datasets():
            return self._tensorflow_dataset._exists() and self._pickle_dataset._exists()

    def _release(self) -> None:
        super()._release()
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate underlying filesystem caches."""
        filepath = get_filepath_str(self._filepath, self._protocol)
        self._fs.invalidate_cache(filepath)

    def _describe(self) -> tp.Dict[str, tp.Any]:
        return {
            "filepath": self._filepath,
            "protocol": self._protocol,
            "tensorflow_dataset_load_args": self._tensorflow_dataset._load_args,
            "pickle_dataset_load_args": self._pickle_dataset._load_args,
            "tensorflow_dataset_save_args": self._tensorflow_dataset._save_args,
            "pickle_dataset_save_args": self._pickle_dataset._save_args,
            "version": self._version,
        }