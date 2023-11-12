import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List

import numpy
import numpy as np
import pandas as pd
from psycopg2.extensions import AsIs, register_adapter

from optimus_core import TagDict

register_adapter(numpy.float32, lambda x: AsIs(x))
register_adapter(numpy.int64, lambda x: AsIs(x))

logger = logging.getLogger(__name__)


def _get_last_actual_value(
    df: pd.DataFrame,
    required_timestamp: str,
    pi_data: pd.DataFrame,
    pi_with_derived_features: pd.DataFrame,
    td: TagDict,
):
    """

    Args:
        df: data for which last values are required, must contain tag column
        required_timestamp: timestamp for which original data is to be fetched
        pi_data: original pi data with timezone set and basic cleaning, non-outlier cleaned
        pi_with_derived_features: pi with derived features data, non-outlier cleaned
        td: tag dictionary to filter apt tags from td

    Returns: actual data for required tags with requested timestamp

    """
    pi_tags = td.select(
        condition=(lambda row: (row["final_recs"] == 1 and row["derived"] == False))
    )

    pi_tags_actual_data = pi_data[pi_data.index == required_timestamp][
        pi_tags
    ].T.reset_index()

    other_tags = td.select(
        condition=(lambda row: (row["final_recs"] == 1 and row["derived"] == True))
    )

    other_tags_actual_data = pi_with_derived_features[
        pi_with_derived_features.index == required_timestamp
    ][other_tags].T.reset_index()

    actual_values = pd.concat(
        [pi_tags_actual_data, other_tags_actual_data], axis=0, ignore_index=True
    )

    actual_values.columns = ["tag", "last_value"]

    recs_with_actual_values = pd.merge(df, actual_values, on="tag", how="left")

    return recs_with_actual_values


def _merge_last_calculated_values(
    recommend_results_translated: pd.DataFrame(),
    recommendations: pd.DataFrame(),
    kpi_last_calculated_values: pd.DataFrame(),
):
    """
    Left Merge to recommendations df the last calculated values
    Args:
        recommend_results_translated
        recommendations

    Returns: Merged df
    """
    last_calculated_values = (
        recommend_results_translated.loc["curr"]
        .drop("timestamp")
        .to_frame()
        .reset_index()
    )
    last_calculated_values.columns = ["tag", "last_calculated_value"]
    recommendations = pd.merge(
        recommendations, last_calculated_values, on="tag", how="left"
    )

    kpi_last_calculated_values = pd.merge(
        kpi_last_calculated_values, last_calculated_values, on="tag", how="left"
    )
    return recommendations, kpi_last_calculated_values


def _filter_recommendations(
    rec_df: pd.DataFrame, actual_col_name: str, rec_col_name: str, td: TagDict
):
    """
    Filter recommendations which are same as actual value and recs which are null
    Args:
        rec_df: Recommendations dataframe
        actual_col_name: name of actual last value column
        rec_col_name: name of recommended column

    Returns: Filtered final recommendations
    """
    td = td.to_frame()
    rec_df = rec_df.merge(
        td[["tag", "unit_precision"]], how="left", left_on="tag", right_on="tag"
    )
    rec_df["actual_rounded"] = rec_df.apply(
        lambda x: round(x[actual_col_name], int(x["unit_precision"])), axis=1
    )
    rec_df["rec_rounded"] = rec_df.apply(
        lambda x: round(x[rec_col_name], int(x["unit_precision"])), axis=1
    )

    rec_df = rec_df[rec_df["actual_rounded"] != rec_df["rec_rounded"]]
    rec_df = rec_df[~rec_df[rec_col_name].isnull()]

    rec_df = rec_df.drop(["actual_rounded", "rec_rounded", "unit_precision"], axis=1)

    return rec_df


def adjust_for_fe_full_circuit(
    recommend_results_translated: Dict,
    upstream_runs: Dict,
    upstream_recommendations: Dict,
    upstream_predictions: Dict,
    downstream_recommendations: pd.DataFrame,
    downstream_predictions: pd.DataFrame,
    kpis_for_ui: List,
    kpis_for_db: List,
    td: TagDict,
    pi_data: pd.DataFrame,
    pi_with_derived_tags: pd.DataFrame,
    baseline_tph,
) -> Dict[str, pd.DataFrame]:
    """
    This function adjusts the final dataframes to be stored in db and to be used on UI for recommendations display
    Args:
        recommend_results_translated: optimization results dict from upstream
        upstream_runs: Dict storing run_id, data_timestamp from upstream
        upstream_recommendations: Dict storing recommendations per control tag from upstream
        upstream_predictions: Predictions dict from optimization pipeline from upstream
        downstream_recommendations: Dict storing recommendations per control tag from downstream
        downstream_predictions: Predictions dict from optimization pipeline from downstream
        kpis_for_ui: KPIs to display on UI
        kpis_for_db: KPIs to store in db (can be different from ones displayed on UI)
        td: Tag dictionary to fetch derived control tags names
        pi_data: Original non-outlier cleaned data coming from pi
        pi_with_derived_tags: Non-outlier cleaned data for derived features
        baseline_tph: Baseline tph value of incoming cluster

    Returns: runs for run_id information, predictions for overall kpis, recommendations at control level dataframes
             and last_calculated_value for important kpis

    """
    ##############################################################################################################
    # For runs table
    # Runs table consists of run_id as id, data current timestamp as data_timestamp and below fields
    ##############################################################################################################
    runs = pd.DataFrame(upstream_runs)
    runs["error_message"] = ""
    # Below refers to run timestamp, data timestamp is separately stored as data_timestamp
    runs["timestamp"] = datetime.now(timezone.utc)

    if len(recommend_results_translated.keys()) > 1:
        logger.error(
            "multiple recommendations, len:%s", len(recommend_results_translated)
        )
        raise Exception("multiple recommendations")
    runs["data_timestamp"] = recommend_results_translated[0].loc["curr", "timestamp"]

    # Last time stamp for fetching actual values may not be same as aggregated frequency timestamp as aggregation
    # happens every 4h however current time could be in between those 4h
    last_timestamp = pi_data.tail(1).index[0]

    # Assigning same run id to kpi_last_calculated_value recommendations
    run_id_values = []
    for x in range(len(kpis_for_db)):
        run_id_values.append(runs["id"][0])

    kpi_last_calculated_values = pd.DataFrame(run_id_values, columns=["run_id"])
    kpi_last_calculated_values["tag"] = pd.Series(kpis_for_db)

    # Assigning same run id to downstream recommendations
    downstream_recommendations["run_id"] = runs["id"][0]
    downstream_recommendations["recommendation_status"] = "Pending"
    downstream_recommendations["comment"] = ""

    # For "recommendations" table
    recommendations = pd.DataFrame(upstream_recommendations)

    recommendations = pd.concat(
        [recommendations, downstream_recommendations], ignore_index=True
    )

    recommendations["expiration_time"] = 60
    recommendations["is_flagged"] = False
    recommendations["accepted_value"] = None
    recommendations["implementation_status"] = 0
    recommendations["recommendation_status"] = "Pending"
    recommendations["initiated_at"] = pd.NaT
    recommendations["initiated_by_user_id"] = None
    recommendations["initiated_by_user_name"] = None

    ##############################################################################################################
    # Get last value at current time stamp of pi and derived features
    ##############################################################################################################

    recommendations = _get_last_actual_value(
        recommendations, last_timestamp, pi_data, pi_with_derived_tags, td
    )

    ##############################################################################################################
    # Get last calculated value -- 4h right aggregated average going in optimizer
    ##############################################################################################################

    recommendations, kpi_last_calculated_values = _merge_last_calculated_values(
        recommend_results_translated[0], recommendations, kpi_last_calculated_values
    )

    ##############################################################################################################
    # Filter recommendations
    ##############################################################################################################

    recommendations = _filter_recommendations(
        rec_df=recommendations,
        actual_col_name="last_value",
        rec_col_name="recommended_value",
        td=td,
    )

    ##############################################################################################################
    # For prediction table
    ##############################################################################################################
    predictions = pd.DataFrame(upstream_predictions)

    # Assigning same run id to downstream predictions (Gold recovery would come from downstream predictions
    downstream_predictions["run_id"] = runs["id"][0]
    predictions = pd.concat([predictions, downstream_predictions], ignore_index=True)

    # TODO: Let's keep tag_id and not rename it by target_id
    predictions.rename(columns={"tag_id": "tag"}, inplace=True)

    new_prediction_kpis = pd.DataFrame(columns=predictions.columns)
    row_num = 0
    for tag in kpis_for_ui:
        new_prediction_kpis.loc[row_num, "run_id"] = runs["id"][0]
        new_prediction_kpis.loc[row_num, "id"] = str(uuid.uuid4())
        new_prediction_kpis.loc[row_num, "tag"] = tag
        new_prediction_kpis.loc[row_num, "actual"] = recommend_results_translated[0][
            tag
        ]["curr"]
        # TODO: Replace baseline value with baseline historical value calculated via baseline pipeline
        # currently set to actual value
        new_prediction_kpis.loc[row_num, "baseline"] = recommend_results_translated[0][
            tag
        ]["curr"]
        new_prediction_kpis.loc[row_num, "optimized"] = recommend_results_translated[0][
            tag
        ]["opt"]
        row_num = row_num + 1

    predictions = pd.concat([predictions, new_prediction_kpis], ignore_index=True)

    predictions["baseline"] = np.where(
        predictions["tag"] == "200_cv_001_weightometer",
        baseline_tph,
        predictions["baseline"],
    )
    predictions["potential_uplift"] = (
        predictions["optimized"] / predictions["baseline"]
    ) - 1
    # TODO: calculation differs for incremental entries, maybe we need a different col
    predictions.loc[
        predictions["tag"].isin(["incremental_gold_produced"]), "potential_uplift"
    ] = (predictions["actual"] - predictions["baseline"])
    # TODO: create new columns in table for below fields
    # predictions["achieved_uplift"] = (predictions["optimized"] / predictions["baseline"]) - 1

    # Objective target_id not required in predictions table, removing it from output
    predictions = predictions[predictions["tag"] != "objective"]

    return dict(
        runs=runs,
        recommendations=recommendations,
        predictions=predictions,
        kpi_last_calculated_value=kpi_last_calculated_values,
    )


def adjust_for_fe_upstream(
    recommend_results_translated: Dict,
    upstream_runs: Dict,
    upstream_recommendations: Dict,
    upstream_predictions: Dict,
    kpis_for_ui: List,
    kpis_for_db: List,
    td: TagDict,
    pi_data: pd.DataFrame,
    pi_with_derived_tags: pd.DataFrame,
    baseline_tph: List,
) -> Dict[str, pd.DataFrame]:
    """
    This function adjusts the final dataframes to be stored in db and to be used on UI for recommendations display
    Args:
        recommend_results_translated: optimization results dict from upstream
        upstream_runs: Dict storing run_id, data_timestamp from upstream
        upstream_recommendations: Dict storing recommendations per control tag from upstream
        upstream_predictions: Predictions dict from optimization pipeline from upstream
        kpis_for_ui: KPIs to display on UI
        kpis_for_db: KPIs to store in db (can be different from ones displayed on UI)
        td: Tag dictionary to fetch derived control tags names
        pi_data: Original non-outlier cleaned data coming from pi
        pi_with_derived_tags: Non-outlier cleaned data for derived features
        baseline_tph: Baseline tph value of incoming cluster

    Returns: runs for run_id information, predictions for overall kpis and recommendations at control level dataframes

    """
    ##############################################################################################################
    # For runs table
    # Runs table consists of run_id as id, data current timestamp as data_timestamp and below fields
    ##############################################################################################################
    runs = pd.DataFrame(upstream_runs)
    runs["error_message"] = ""
    # Below refers to run timestamp, data timestamp is separately stored as data_timestamp
    runs["timestamp"] = datetime.now(timezone.utc)

    if len(recommend_results_translated.keys()) > 1:
        logger.error(
            "multiple recommendations, len:%s", len(recommend_results_translated)
        )
        raise Exception("multiple recommendations")
    runs["data_timestamp"] = recommend_results_translated[0].loc["curr", "timestamp"]

    # Last time stamp for fetching actual values may not be same as aggregated frequency timestamp as aggregation
    # happens every 4h however current time could be in between those 4h
    last_timestamp = pi_data.tail(1).index[0]

    # Assigning same run id to kpi_last_calculated_value recommendations
    run_id_values = []
    for x in range(len(kpis_for_db)):
        run_id_values.append(runs["id"][0])

    kpi_last_calculated_values = pd.DataFrame(run_id_values, columns=["run_id"])
    kpi_last_calculated_values["tag"] = pd.Series(kpis_for_db)

    # For "recommendations" table
    recommendations = pd.DataFrame(upstream_recommendations)

    recommendations["expiration_time"] = 60
    recommendations["is_flagged"] = False
    recommendations["accepted_value"] = None
    recommendations["implementation_status"] = 0
    recommendations["recommendation_status"] = "Pending"
    recommendations["initiated_at"] = pd.NaT
    recommendations["initiated_by_user_id"] = None
    recommendations["initiated_by_user_name"] = None

    ##############################################################################################################
    # Get last value at current time stamp of pi and derived features
    ##############################################################################################################

    recommendations = _get_last_actual_value(
        recommendations, last_timestamp, pi_data, pi_with_derived_tags, td
    )

    ##############################################################################################################
    # Get last calculated value -- 4h right aggregated average going in optimizer
    ##############################################################################################################

    recommendations, kpi_last_calculated_values = _merge_last_calculated_values(
        recommend_results_translated[0], recommendations, kpi_last_calculated_values
    )

    ##############################################################################################################
    # Filter recommendations
    ##############################################################################################################

    recommendations = _filter_recommendations(
        rec_df=recommendations,
        actual_col_name="last_value",
        rec_col_name="recommended_value",
        td=td,
    )

    ##############################################################################################################
    # For prediction table
    ##############################################################################################################
    predictions = pd.DataFrame(upstream_predictions)
    # TODO: Let's keep tag_id and not rename it by target_id
    predictions.rename(columns={"tag_id": "tag"}, inplace=True)

    new_prediction_kpis = pd.DataFrame(columns=predictions.columns)
    row_num = 0
    for tag in kpis_for_ui:
        new_prediction_kpis.loc[row_num, "run_id"] = runs["id"][0]
        new_prediction_kpis.loc[row_num, "id"] = str(uuid.uuid4())
        new_prediction_kpis.loc[row_num, "tag"] = tag
        new_prediction_kpis.loc[row_num, "actual"] = recommend_results_translated[0][
            tag
        ]["curr"]
        # TODO: Replace baseline value with baseline historical value calculated via baseline pipeline
        # currently set to actual value
        new_prediction_kpis.loc[row_num, "baseline"] = recommend_results_translated[0][
            tag
        ]["curr"]
        new_prediction_kpis.loc[row_num, "optimized"] = recommend_results_translated[0][
            tag
        ]["opt"]
        row_num = row_num + 1

    predictions = pd.concat([predictions, new_prediction_kpis], ignore_index=True)

    predictions["baseline"] = np.where(
        predictions["tag"] == "200_cv_001_weightometer",
        baseline_tph,
        predictions["baseline"],
    )
    predictions["potential_uplift"] = (
        predictions["optimized"] / predictions["baseline"]
    ) - 1

    # TODO: create new columns in table for below fields
    # predictions["achieved_uplift"] = (predictions["optimized"] / predictions["baseline"]) - 1

    # Objective target_id not required in predictions table, removing it from output
    predictions = predictions[predictions["tag"] != "objective"]

    return dict(
        runs=runs,
        recommendations=recommendations,
        predictions=predictions,
        kpi_last_calculated_value=kpi_last_calculated_values,
    )
