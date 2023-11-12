from kedro.pipeline import Pipeline, node, pipeline

from alchemy.pipelines.data_engineering.ingestion.ingestion_nodes import (
    ingest_pi_data,
    ingest_raw_pi_data,
)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            # node(
            #     func=ingest_raw_pi_data,
            #     inputs="pi_data",
            #     outputs="raw_pi_data_ingested",
            #     name="raw_ingest_pi_data",
            # ),
            node(
                func=ingest_pi_data,
                inputs=["raw_pi_data_ingested", "params:pi_data"],
                outputs="pi_data_ingested",
                name="ingest_pi_data",
            ),
        ],
        tags="ingestion",
    )
