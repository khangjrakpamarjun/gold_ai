"""Project pipelines."""
import warnings
from typing import Dict

import pandas as pd
from kedro.pipeline import Pipeline

from alchemy.pipelines.baseline import baseline_pipeline as bp
from alchemy.pipelines.data_engineering import data_engineering_pipeline as de
from alchemy.pipelines.data_science import data_science_pipeline as ds
from alchemy.pipelines.optimization.optimization import optimization_pipeline as op
from alchemy.pipelines.optimization.optimization.optimization_pipeline import (
    create_downstream_pipeline,
    create_export_cf_pipeline,
    create_export_recs_pipeline,
    create_upstream_pipeline,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Importing display")


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's data engineering, data science and optimization pipelines

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    de_pipe = de.create_pipeline()
    ds_pipe = ds.create_pipeline()

    retrain_pipe = de_pipe + ds_pipe
    baseline_pipe = de_pipe + bp.create_baselines_pipeline()

    # TODO: Refactor below function calls using partial wrapper or some better methods

    cf_pipe = op.create_recommendations_pipeline(
        env="base",
        upstream_pipeline=create_upstream_pipeline(testing=False),
        downstream_pipeline=create_downstream_pipeline(testing=False),
        export_recs_pipeline=create_export_cf_pipeline(),
    )

    live_full_pipe = de_pipe + op.create_recommendations_pipeline(
        env="live",
        upstream_pipeline=create_upstream_pipeline(testing=False),
        downstream_pipeline=create_downstream_pipeline(testing=False),
        export_recs_pipeline=create_export_recs_pipeline(full_circuit=True),
    )

    live_upstream_pipe = de_pipe + op.create_upstream_export_pipeline(
        env="live",
        upstream_pipeline=create_upstream_pipeline(testing=False),
        export_recs_pipeline=create_export_recs_pipeline(full_circuit=False),
    )

    test_pipe = op.create_recommendations_pipeline(
        env="base",
        upstream_pipeline=create_upstream_pipeline(testing=True),
        downstream_pipeline=create_downstream_pipeline(testing=True),
        export_recs_pipeline=create_export_cf_pipeline(),
    )

    pipeline_dict = {
        "__default__": cf_pipe,
        "de_pipe": de_pipe,
        "ds_pipe": ds_pipe,
        "baseline_pipe": baseline_pipe,  # baseline analysis for tph
        "retrain_pipe": retrain_pipe,
        "live_upstream_pipe": live_upstream_pipe,  # For live recomm use: --env live
        "live_full_pipe": live_full_pipe,
        "cf_pipe": cf_pipe,  # Counterfactual analysis
        "test_pipe": test_pipe,  # Testing pipeline
    }
    return pipeline_dict
