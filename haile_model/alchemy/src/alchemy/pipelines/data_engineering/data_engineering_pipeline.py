import logging

from kedro.pipeline import Pipeline

from alchemy.pipelines.data_engineering.feature_engineering import (
    feature_engineering_pipeline as fe,
)
from alchemy.pipelines.data_engineering.ingestion import ingestion_pipeline as di
from alchemy.pipelines.data_engineering.post_processing import (
    post_processing_pipeline as dpost,
)
from alchemy.pipelines.data_engineering.pre_processing import (
    pre_processing_pipeline as dpre,
)
from alchemy.pipelines.data_engineering.prepare_test_data import (
    prepare_test_data_pipeline as test,
)

logger = logging.getLogger(__name__)


def create_pipeline() -> Pipeline:
    data_ingestion_pipe = di.create_pipeline()
    data_preprocessing_pipe = dpre.create_pipeline()
    feature_engineering_pipe = fe.create_pipeline()
    data_postprocessing_pipe = dpost.create_pipeline()
    prepare_test_data_pipe = test.create_pipeline()

    pipelines = (
        data_ingestion_pipe
        + data_preprocessing_pipe
        + feature_engineering_pipe
        + data_postprocessing_pipe
        + prepare_test_data_pipe
    )

    return pipelines
