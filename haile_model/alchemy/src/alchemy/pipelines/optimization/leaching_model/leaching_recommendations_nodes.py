"""
This is a boilerplate pipeline 'leaching'
generated using Kedro 0.18.7
"""
import logging
import uuid
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from optimus_core import TagDict
from optimus_core.tag_dict import TagDict

logger = logging.getLogger(__name__)


def impute_na_in_downstream_features(
    test_data: pd.DataFrame,
    median_values_for_tags: pd.DataFrame,
    td: TagDict,
    params: Dict,
) -> pd.DataFrame:
    """
    Impute NAs present in test data with median values from train data saved in table
    Args:
        test_data: shift level test data
        median_values_for_tags: table containing the median value of downstream features
        td: tag dictionary
        params: dictionary containing parameters

    Returns:
    test data with NAs filled with median
    """
    feature_downstream_cluster = params["feature_downstream_cluster"]

    feature_operating_modes = params["feature_operating_modes"]
    control_variables = td.select(
        condition=(
            lambda row: (row["tag_type"] == "control" and row[feature_operating_modes])
        )
    )
    ore_variables = td.select(condition=(lambda row: (row[feature_downstream_cluster])))
    leaching_vars = control_variables + ore_variables
    for i in range(len(leaching_vars)):
        var = leaching_vars[i]
        median_val = median_values_for_tags[var]

        for j in range(test_data.shape[0]):
            if pd.isna(test_data[var][j]):
                test_data[var][j] = median_val
                msg = (
                    f"feature {var} is NA in the test data. Instead median value"
                    f" {median_val} from train is returned "
                )
                logger.warning(msg)

    return test_data


def drop_null(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model_features_first_level_cluster: List,
    model_features_second_level_cluster: List,
    params: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fills NAs in train data with median of each column. Fills NAs of test data with
    median of respective cols from train data. Drops rows with NA values in recovery
    column in train and test data.
    Args:
        train_data: Shift level train data with clusters and best operating modes
        test_data: Shift level new data
        params: dictionary containing params
        model_features_first_level_cluster: features(6x) used in training the ore
        characteristics clusters
        model_features_second_level_cluster: features used in training the operating
        modes cluster
    Returns: Returns shift level train and test data with nulls removed

    """
    train_data[model_features_second_level_cluster] = train_data[
        model_features_second_level_cluster
    ].fillna(train_data[model_features_second_level_cluster].median())
    test_data[model_features_second_level_cluster] = test_data[
        model_features_second_level_cluster
    ].fillna(train_data.median())
    train_data[model_features_first_level_cluster].fillna(
        train_data[model_features_first_level_cluster].median(), inplace=True
    )
    test_data[model_features_first_level_cluster] = test_data[
        model_features_first_level_cluster
    ].fillna(train_data.median())
    train_data = train_data.dropna(subset=params["recovery"], axis=0)
    test_data = test_data.dropna(subset=params["recovery"], axis=0)
    return train_data, test_data


def classify_ore_cluster(
    test_data: pd.DataFrame,
    first_cluster_trained_model: Dict,
    model_features_first_level_cluster: List,
) -> pd.DataFrame:
    """
    Classify the ore cluster for new shifts using the saved model trained on train
    data and maps its best operating mode from the saved mapping file
    Args:
        test_data: Shift level data without null
        first_cluster_trained_model: model object trained on train data
        for first level cluster
        model_features_first_level_cluster: features(6x) used in training
        the ore characteristics clusters

    Returns: dataframe with ore cluster classified along with ore characteristics tags

    """
    first_level_cluster_trained_model = first_cluster_trained_model
    x_test_index = test_data["timestamp"]
    test_data_upstream = test_data[model_features_first_level_cluster]
    target = "cluster_kmeans"
    x_test = test_data_upstream
    y_prediction = first_level_cluster_trained_model.predict(x_test)
    x_test[target] = y_prediction
    x_test.index = x_test_index
    return x_test


def map_operating_modes_to_ore_clusters(
    test_data_without_nulls_all_tags: pd.DataFrame,
    test_data: pd.DataFrame,
    params: Dict,
    best_modes_per_cluster: Dict,
    model_features_second_level_cluster: List,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Map the best operating mode from the saved mapping file for each
     of the classified  ore cluster
    Args:
        test_data_without_nulls_all_tags: Shift level data containing
        all tags
        test_data: Shift level data with ore clusters along with ore
        characteristics tags
        params: dict containing parameters
        best_modes_per_cluster: dict containing the mapping of clusters
        with their best operating modes.
        model_features_second_level_cluster: features used in training
        the operating modes cluster

    Returns: Shift level dataframes with clusters and best operating
    modes containing ore characteristics cluster tags
    and all tags used in modelling the two clusters respectively

    """
    x_test_index = test_data.index
    x_test = test_data
    cluster_kmeans = params["cluster_col_name"]
    operating_modes = params["operating_modes"]
    df = pd.DataFrame(best_modes_per_cluster, index=[0]).T
    df.reset_index(inplace=True)
    df.columns = [cluster_kmeans, operating_modes]
    df[cluster_kmeans] = df[cluster_kmeans].astype(str).astype(int)
    x_test = x_test.merge(df, on=cluster_kmeans, how="left")
    x_test.index = x_test_index
    test_x_clusters_opm = test_data_without_nulls_all_tags[
        model_features_second_level_cluster
    ]
    test_x_clusters_opm.index = x_test_index
    test_data_all_tags_with_clusters = test_x_clusters_opm.join(x_test)
    test_data_all_tags_with_clusters.index = x_test_index
    test_data_without_nulls_all_tags.index = test_data_without_nulls_all_tags[
        "timestamp"
    ]
    test_data_without_nulls_all_tags = test_data_without_nulls_all_tags.drop(
        "timestamp", axis=1
    )
    test_data_all_tags_with_clusters = test_data_all_tags_with_clusters.join(
        test_data_without_nulls_all_tags[params["recovery"]]
    )
    test_data_all_tags_with_clusters.index = x_test_index
    test_data_ore_tags_with_clusters = x_test
    return test_data_ore_tags_with_clusters, test_data_all_tags_with_clusters


def classify_operating_modes(
    test_data: pd.DataFrame,
    second_cluster_trained_models: List,
    model_features_second_level_cluster: List,
    model_features_second_level_cluster_dict: Dict,
) -> pd.DataFrame:
    """
    Classify the operating modes for each ore clusters using the
    saved models which was trained on the train data

    Args:
        test_data: test data containing all the tags with predicted
        ore cluster and mapped best operating modes
        second_cluster_trained_models: trained model for classifying
        the operating modes cluster
        model_features_second_level_cluster: features used in
        training the operating modes cluster
        model_features_second_level_cluster_dict: dict containing the
        list of features used for each clusters
    Returns: returns the test data with all tags

    """
    unique_clusters = test_data["cluster_kmeans"].unique().tolist()
    combine_test_opm_predicted = pd.DataFrame()
    for cluster in range(len(unique_clusters)):
        cluster = int(unique_clusters[cluster])
        test_data_modes = test_data[(test_data["cluster_kmeans"] == cluster)]
        index_test = test_data_modes.index
        model_features_second_level_cluster = model_features_second_level_cluster_dict[
            str(cluster)
        ]
        second_level_cluster_trained_model = second_cluster_trained_models[str(cluster)]
        test_data_modes = test_data_modes[model_features_second_level_cluster]
        y_prediction = second_level_cluster_trained_model.predict(test_data_modes)
        test_data_modes["operating_modes_predicted"] = y_prediction
        test_data_modes.index = index_test
        combine_test_opm_predicted = pd.concat(
            [combine_test_opm_predicted, test_data_modes]
        )
    test_data_with_ore_clusters = test_data.join(
        combine_test_opm_predicted["operating_modes_predicted"]
    )

    return test_data_with_ore_clusters


def get_historical_baseline_recovery(
    test_data_with_flc_and_slc,
    test_data: pd.DataFrame,
    td: TagDict,
    params: Dict,
    first_cluster_trained_model: List,
    second_cluster_trained_models: List,
    model_features_first_level_cluster: List,
    model_features_second_level_cluster_dict: List,
    baseline_historic_upstream_and_downstream: pd.DataFrame,
) -> Tuple:
    """
    Get the hist baseline in recommendations tables
    Args:
        test_data_with_flc_and_slc: test data containing clusters
        test_data:test data containing all the tags with predicted
        ore cluster and mapped best operating modes
        td:tag dict
        params: dict containing parameters
        first_cluster_trained_model: first cluster trained baseline model
        second_cluster_trained_models: second cluster trained baseline model
        model_features_first_level_cluster: features used in first cluster trained baseline model
        model_features_second_level_cluster_dict: features used in second cluster trained baseline model
        baseline_historic_upstream_and_downstream: data containing the upstream and downstream baselines

    Returns:
    returns the test data with historical baseline recovery along with predicted clusters using baseline models
    """
    first_level_cluster_trained_model = first_cluster_trained_model
    x_test_index = test_data["timestamp"]
    test_data_upstream = test_data[model_features_first_level_cluster]
    target = "cluster_kmeans_hist_baseline"
    x_test = test_data_upstream
    y_prediction = first_level_cluster_trained_model.predict(x_test)
    x_test[target] = y_prediction
    x_test.index = x_test_index
    test_data.index = test_data["timestamp"]
    test_data_slc = test_data.join(x_test["cluster_kmeans_hist_baseline"])
    unique_clusters = test_data_slc["cluster_kmeans_hist_baseline"].unique().tolist()
    combine_test_opm_predicted = pd.DataFrame()
    for cluster in range(len(unique_clusters)):
        cluster = int(unique_clusters[cluster])
        test_data_modes = test_data_slc[
            (test_data_slc["cluster_kmeans_hist_baseline"] == cluster)
        ]
        index_test = test_data_modes.index
        model_features_second_level_cluster = model_features_second_level_cluster_dict[
            str(cluster)
        ]
        second_level_cluster_trained_model = second_cluster_trained_models[str(cluster)]
        test_data_modes = test_data_modes[model_features_second_level_cluster]
        y_prediction = second_level_cluster_trained_model.predict(test_data_modes)
        test_data_modes["operating_modes_hist_baseline"] = y_prediction
        test_data_modes.index = index_test
        combine_test_opm_predicted = pd.concat(
            [combine_test_opm_predicted, test_data_modes]
        )
    test_data_with_ore_clusters = test_data_slc.join(
        combine_test_opm_predicted["operating_modes_hist_baseline"]
    )
    test_data_with_ore_clusters["timestamp"] = test_data_with_ore_clusters.index
    test_data_with_ore_clusters = test_data_with_ore_clusters.drop("timestamp", axis=1)

    unique_cluster_rec = unique_clusters
    test_data_modes = test_data_with_ore_clusters
    test_data_modes["recovery_hist_baseline"] = np.nan
    # Select records for predicted recovery calculation at ore
    # characteristic and operating mode level and use
    # operating_modes_predicted
    base = baseline_historic_upstream_and_downstream
    base = base[base["tag_id"] == "gold_recovery"]
    base["operating_modes"] = base["operating_modes"].astype(str).astype(int)
    baseline_historic_upstream_and_downstream = base
    for cluster in range(len(unique_cluster_rec)):
        cluster = int(unique_cluster_rec[cluster])
        unique_operating_modes_hist_baseline = test_data_modes[
            test_data_modes["cluster_kmeans_hist_baseline"] == cluster
        ]["operating_modes_hist_baseline"].unique()
        for operating_mode in range(len(unique_operating_modes_hist_baseline)):
            operating_mode = int(unique_operating_modes_hist_baseline[operating_mode])
            value = baseline_historic_upstream_and_downstream[
                (
                    baseline_historic_upstream_and_downstream["operating_modes"]
                    == operating_mode
                )
                & (baseline_historic_upstream_and_downstream["cluster"] == cluster)
            ]["baseline"]
            test_data_modes["recovery_hist_baseline"] = np.where(
                (test_data_modes["cluster_kmeans_hist_baseline"] == cluster)
                & (test_data_modes["operating_modes_hist_baseline"] == operating_mode),
                value,
                test_data_modes["recovery_hist_baseline"],
            )
    test_data_clusters_recovery_hist_baseline = test_data_with_flc_and_slc.join(
        test_data_modes[
            [
                "operating_modes_hist_baseline",
                "cluster_kmeans_hist_baseline",
                "recovery_hist_baseline",
            ]
        ]
    )
    return (
        test_data_clusters_recovery_hist_baseline,
        test_data_clusters_recovery_hist_baseline,
    )


def _get_opt_min_max(
    df: pd.DataFrame, control_variables: List, td: TagDict, params: Dict
):
    """
    Enforce min max opt bounds in the recommendations of control variables for test data
    Args:
        df: test data
        control_variables: list of controls variables
        td: tag dictionary
        params: dict containing dictionary

    Returns:
    test data with min max opt bounds enforced in the recommendations of control variable
    """
    feature_list = df[control_variables].columns
    for feature in range(len(feature_list)):
        feature = feature_list[feature]
        op_min = td[feature]["op_min"]
        op_max = td[feature]["op_max"]
        for row in range(df.shape[0]):
            if df[feature][row] > op_max:
                msg = (
                    f"Optimized value of {feature} is {df[feature][row]} which is more"
                    f" than op_max value {op_max}. {op_max} will be returned instead"
                )
                df[feature][row] = op_max
                logger.warning(msg)
            if df[feature][row] < op_min:
                msg = (
                    f"Optimized value of {feature} is {df[feature][row]} which is less"
                    f" than op_min value {op_min}. {op_min} will be returned instead"
                )
                df[feature][row] = op_min
                logger.warning(msg)
    return df


def _get_predicted_recovery(
    test_data_modes: pd.DataFrame,
    unique_cluster_rec: List,
    all_operating_mode_per_cluster: Dict,
):
    """
    Adds "predicted_recovey" as the median for the predicted ore and mode cluster

    Args:
        test_data_modes: df to enrich
        unique_cluster_rec: clusters
        all_operating_mode_per_cluster: Dict with cluster and mode statistical values

    Returns: df with added column "predicted_recovery"
    """
    test_data_modes["predicted_recovery"] = np.nan
    for cluster in unique_cluster_rec:
        # get modes
        unique_operating_mode_rec_pred = test_data_modes[
            test_data_modes["cluster_kmeans"] == cluster
        ]["operating_modes_predicted"].unique()
        unique_operating_mode_rec_pred = [
            int(mode) for mode in unique_operating_mode_rec_pred
        ]

        for operating_mode in unique_operating_mode_rec_pred:
            median_value = all_operating_mode_per_cluster[cluster]["50%"][
                operating_mode
            ]
            test_data_modes["predicted_recovery"] = np.where(
                (test_data_modes["cluster_kmeans"] == cluster)
                & (test_data_modes["operating_modes_predicted"] == operating_mode),
                median_value,
                test_data_modes["predicted_recovery"],
            )
    return test_data_modes


def _get_rec_stats(
    kneighbors: int,
    train_data_cluster: pd.DataFrame,
    op_mode_lower_quantile: float,
    op_mode_upper_quantile: float,
    recovery_column: str,
) -> (pd.Series, pd.Series, pd.Series):
    """
    Calculates median and quartiles from top NearestNeighbors

    Args:
        kneighbors: selected neighbor
        train_data_cluster: clustered data
        op_mode_lower_quantile:
        op_mode_upper_quantile:
        recovery_column:

    Returns: median_value, lower_percentile, upper_percentile for the selected neighbor
    """
    # select top nearest neighbors from train data
    n_neighbors = train_data_cluster.reset_index(drop=True).loc[kneighbors]

    # filter the rows with top 50 % recovery
    median_value = n_neighbors[recovery_column].median()
    top_neighbors = n_neighbors[n_neighbors[recovery_column] >= median_value]

    # calculate the median and quartiles of the above rows
    median_value = top_neighbors.median()
    lower_percentile = top_neighbors.quantile(op_mode_lower_quantile)
    upper_percentile = top_neighbors.quantile(op_mode_upper_quantile)

    return median_value, lower_percentile, upper_percentile


def _get_test_neighbors(
    train_data_cluster: pd.DataFrame,
    test_data_cluster: pd.DataFrame,
    n_neighbors: int,
):
    """
    Get the NN for the test dataset

    Args:
        train_data_cluster: clustered train data
        train_data_cluster: clustered test data
        n_neighbors: number of neighbors to keep

    Returns: neighbors for test data
    """
    # scale data
    scaler = MinMaxScaler()  # TODO: must store the scaler in a unique location
    scaler.fit(train_data_cluster)
    train_data_cluster_scaled = scaler.transform(train_data_cluster)
    test_data_cluster_scaled = scaler.transform(test_data_cluster)

    # Take the value from the nearest n_neighbors NEIGHBORS
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors.fit(train_data_cluster_scaled)
    neighbors_test = neighbors.kneighbors(
        test_data_cluster_scaled, return_distance=False
    )
    return neighbors_test


def _get_processed_recs(
    rec_median: pd.DataFrame,
    test_data_modes: pd.DataFrame,
    control_variables: List,
    timestamp: str,
):
    """
    Process recommendation dataframe for output

    Args:
        rec_median: recs dataframe
        test_data_modes: operating modes for test data
        control_variables: list of control variables
        timestamp: timestamp column

    Returns:
        recommendations_live: processed recs
        recommendations_live_timestamp: processed recs with timestamp
        recommendations_downstream: downstream recs
    """
    # select control_cols
    recommendations_live = rec_median.copy()
    recommendations_live = recommendations_live[control_variables]

    # keep df with timestamp
    recommendations_live_timestamp = recommendations_live.copy()
    recommendations_live_timestamp[timestamp] = test_data_modes.index
    recommendations_live_timestamp = recommendations_live_timestamp[
        [timestamp] + control_variables
    ]
    recommendations_controls = recommendations_live_timestamp.copy()

    # transpose dfs
    recommendations_live = recommendations_live.tail(1).T
    recommendations_live = recommendations_live.reset_index()
    recommendations_live.columns = ["tag", "recommended_value"]
    recommendations_live_timestamp = recommendations_live_timestamp.tail(1).T
    recommendations_live_timestamp = recommendations_live_timestamp.reset_index()
    recommendations_live_timestamp.columns = ["tag", "recommended_value"]

    # add optimized suffix
    recommendations_controls = recommendations_controls.add_suffix("_optimized")
    recommendations_controls.rename(
        columns={f"{timestamp}_optimized": timestamp}, inplace=True
    )

    # add an ID
    recommendations_live["id"] = [
        str(uuid.uuid4()) for _ in range(recommendations_live.shape[0])
    ]

    # PREPROCESS --------------------------------------
    test_data_modes_downstream = test_data_modes.copy()
    test_data_modes_downstream = test_data_modes_downstream[control_variables]
    test_data_modes_downstream.insert(
        loc=0, column=timestamp, value=test_data_modes.index
    )
    test_data_modes_downstream = test_data_modes_downstream.add_suffix("_actual")
    recommendations_downstream = pd.merge(
        test_data_modes_downstream, recommendations_controls, on=timestamp, how="inner"
    )
    dupli_cols = recommendations_downstream.columns[
        recommendations_downstream.columns.duplicated()
    ]
    recommendations_downstream.drop(columns=dupli_cols, inplace=True)
    recommendations_downstream.drop(columns=f"{timestamp}_actual", inplace=True)
    recommendations_downstream = recommendations_downstream[
        sorted(recommendations_downstream.columns)
    ]
    recommendations_downstream.insert(
        0, timestamp, recommendations_downstream.pop(timestamp)
    )

    return (
        recommendations_live,
        recommendations_live_timestamp,
        recommendations_downstream,
    )


def _get_pred_and_rec(
    rec_median: pd.DataFrame,
    test_data_modes: pd.DataFrame,
    control_variables: List,
    timestamp: str,
    baseline: str,
    recovery_column: str,
    incremental_gold_produced: pd.DataFrame,
):
    """
    Computes predictions and recommendations

    Args:
        rec_median: recs dataframe
        test_data_modes: operating modes for test data
        control_variables: list of control variables
        timestamp: timestamp column
        baseline: baseline column
        recovery_column: recovert column

    Returns:
        prediction: gold_recovery predictions
        recommendations: recs merged with test dataset and processed
    """
    # PREPROCESS ---------------------------------------------
    rec_median = rec_median[control_variables + [recovery_column]]
    rec_median = rec_median.add_suffix("_optimized")
    rec_median.insert(loc=0, column=timestamp, value=test_data_modes.index)
    test_data_modes = test_data_modes.reset_index().rename(
        columns={test_data_modes.index.name: timestamp}
    )

    # MERGE --------------------------------------------------
    recommendations = pd.merge(test_data_modes, rec_median, on=timestamp, how="inner")
    duplicate_cols = recommendations.columns[recommendations.columns.duplicated()]
    recommendations.drop(columns=duplicate_cols, inplace=True)
    recommendations = recommendations.reindex(sorted(recommendations.columns), axis=1)

    # ADD COLS ---------------------------------------------------------------
    recommendations["recovery_uplift_%"] = (
        (recommendations["recovery_optimized"] - recommendations["recovery_actual"])
        * 100
        / recommendations["recovery_actual"]
    )

    # RENAMING ---------------------------------------------------------------
    selected_cols = [
        "operating_modes_hist_baseline",
        "cluster_kmeans_hist_baseline",
        "operating_modes_predicted",
        # "operating_modes_predicted", # Repeated column name. In original code only five cols were renamed
        "operating_modes_from_mapping",
        "ore_cluster_predicted",
        timestamp,
    ]
    for col in selected_cols:
        recommendations.insert(0, col, recommendations.pop(col))

    # OUTPUT ---------------------------------------------------------------
    pred_data = [
        [
            "gold_recovery",
            recommendations["recovery_actual"].iloc[-1],
            recommendations[baseline].iloc[-1],
            recommendations["recovery_optimized"].iloc[-1],
            str(uuid.uuid4()),
        ],
        [
            "incremental_gold_produced",
            incremental_gold_produced["actual_increment"].values[0],
            incremental_gold_produced["baseline_increment"].values[0],
            incremental_gold_produced["optimized_increment"].values[0],
            str(uuid.uuid4()),
        ],
    ]
    pred_cols = ["tag_id", "actual", "baseline", "optimized", "id"]
    prediction = pd.DataFrame(pred_data, columns=pred_cols)

    return prediction, recommendations


def _rename_test_cols(test_data_modes):
    """
    Rename columns for final output

    Args:
        test_data_modes: operating modes for test data

    Returns: test_data_modes with renamed columns
    """
    test_data_modes = test_data_modes.add_suffix("_actual")
    test_data_modes.rename(
        columns={
            "predicted_recovery_actual": "recovery_predicted",
            "cluster_kmeans_actual": "ore_cluster_predicted",
            "operating_modes_actual": "operating_modes_from_mapping",
            "operating_modes_predicted_actual": "operating_modes_predicted",
            "regrind_product_gold_grade_actual": "regrind_product_gold_grade",
            "regrind_product_sulphide_sulphur_actual": (
                "regrind_product_sulphide_sulphur"
            ),
            "rougher_tails_gold_grade_actual": "rougher_tails_gold_grade",
            "rougher_tails_sulphur_grade_actual": "rougher_tails_sulphur_grade",
            "regrind_product_p80_actual": "regrind_product_p80",
            "primary_cyclone_overflow_p80_actual": "primary_cyclone_overflow_p80",
            "cluster_kmeans_hist_baseline_actual": "cluster_kmeans_hist_baseline",
            "operating_modes_hist_baseline_actual": "operating_modes_hist_baseline",
            "recovery_hist_baseline_actual": "recovery_hist_baseline",
        },
        inplace=True,
    )
    return test_data_modes


def dict_keys_to_int(dict_in):
    """convert keys of dict to int"""
    return {int(k): v for k, v in dict_in.items()}


def get_downstream_recommendations(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    td: TagDict,
    params: Dict,
    best_modes_per_cluster: Dict,
    model_features_second_level_cluster_dict: Dict,
    all_operating_mode_per_cluster: Dict,
    incremental_gold_produced: pd.DataFrame,
    areas_to_optimize: List = None,
) -> tuple:
    """
    Recommends the optimal operating modes of the process variables with
    a min and max value

    Args:
        train_data: data on which the model is to be trained
        test_data: the new data for which recommendations are to be made
        td: tag dictionary
        params: dictionary containing the parameters used in modeling
        best_modes_per_cluster: dict the clusters and their best operating modes
        model_features_second_level_cluster_dict: features used in training the operating modes of each clusters
        all_operating_mode_per_cluster: the operating modes cluster of all clusters
        areas_to_optimize: Areas to optimize

    Returns: the dataframe with shift level recommendations

    """
    # INPUTS -----------------------------------------------------------------
    recovery_column = params["recovery"]
    control_variables = td.select(
        condition=(
            lambda row: row["tag_type"] == "control"
            and row["area"] in areas_to_optimize
        )
    )
    model_features_second_level_cluster_dict = dict_keys_to_int(
        model_features_second_level_cluster_dict
    )
    best_modes_per_cluster = dict_keys_to_int(best_modes_per_cluster)
    all_operating_mode_per_cluster = dict_keys_to_int(all_operating_mode_per_cluster)

    # PREPROCESS -----------------------------------------------------------------
    train_data_modes = train_data.dropna(subset=recovery_column, axis=0)
    train_data_modes["operating_modes"] = train_data_modes["operating_modes"].astype(
        int
    )

    test_data_modes = test_data.dropna(subset=recovery_column, axis=0)
    test_data_modes["operating_modes"] = test_data_modes["operating_modes"].astype(int)
    test_data_modes["cluster_kmeans"] = test_data_modes["cluster_kmeans"].astype(int)

    unique_cluster_rec = list(test_data_modes["cluster_kmeans"].unique())

    # GET RECOVERY FOR TEST -----------------------------------------------------------------
    test_data_modes = _get_predicted_recovery(
        test_data_modes, unique_cluster_rec, all_operating_mode_per_cluster
    )

    # GET RECS -----------------------------------------------------------------
    rec_median = pd.DataFrame()
    rec_low_perc = pd.DataFrame()
    rec_up_perc = pd.DataFrame()
    for cluster in unique_cluster_rec:
        # INPUTS ---------------------------------------------------------------
        cols_to_filter = model_features_second_level_cluster_dict[cluster] + [
            recovery_column
        ]

        # PREPROCESS ---------------------------------------------------------------
        train_data_cluster = train_data_modes[
            (train_data_modes["cluster_kmeans"] == cluster)
            & (train_data_modes["operating_modes"] == best_modes_per_cluster[cluster])
        ]
        train_data_cluster = train_data_cluster[cols_to_filter]

        test_data_cluster = test_data_modes[
            test_data_modes["cluster_kmeans"] == cluster
        ]
        test_data_cluster = test_data_cluster[cols_to_filter]

        # GET Neighbors ---------------------------------------------------------------
        neighbors_test = _get_test_neighbors(
            train_data_cluster, test_data_cluster, params["n_neighbors"]
        )

        for kneighbors in neighbors_test:
            median_value, lower_percentile, upper_percentile = _get_rec_stats(
                kneighbors,
                train_data_cluster,
                params["op_mode_lower_quantile"],
                params["op_mode_upper_quantile"],
                recovery_column,
            )
            rec_median = pd.concat(
                [rec_median, median_value.to_frame().T], ignore_index=True
            )
            rec_low_perc = pd.concat(
                [rec_low_perc, lower_percentile.to_frame().T], ignore_index=True
            )
            rec_up_perc = pd.concat(
                [rec_up_perc, upper_percentile.to_frame().T], ignore_index=True
            )
    rec_median = _get_opt_min_max(rec_median, control_variables, td, params)

    (
        recommendations_live,
        recommendations_live_timestamp,
        recommendations_downstream,
    ) = _get_processed_recs(
        rec_median,
        test_data_modes,
        control_variables,
        params["timestamp"],
    )

    test_data_modes = _rename_test_cols(test_data_modes)

    prediction, recommendations = _get_pred_and_rec(
        rec_median,
        test_data_modes,
        control_variables,
        params["timestamp"],
        params["baseline"],
        recovery_column,
        incremental_gold_produced,
    )

    return (
        recommendations,
        recommendations_live,
        recommendations_live_timestamp,
        recommendations_downstream,
        prediction,
    )


def get_incremental_gold_produced(
    data: pd.DataFrame,
    data_upstream_optimized: pd.DataFrame,
    baseline_tph: pd.Series,
    baseline_recovery: pd.DataFrame,
    params: Dict,
) -> pd.DataFrame:
    """
    Calculates incremental_gold_produced
    actual_increment = head_grade * recovery_actual * throughput_actual / oz_to_g
    baseline_increment = head_grade * recovery_baseline * throughput_baseline / oz_to_g

    Args:
        data: primary data
        data_upstream_optimized: optimized data from given shift
        baseline_tph: throughput baseline value for current cluster
        baseline_recovery: baseline recovery for current cluster
        params:

    Returns:
        df_incremental_gold_produced: df with actual_increment, baseline_increment, increment_uplift
    """
    # TODO: hard fix by taking last row, must unify format/timestamps btw aggregation and latest data
    actual_increment = data["incremental_gold_produced"].iloc[-1]
    actual_head_gold_grade = data["head_gold_grade"].iloc[-1]
    recovery_baseline = baseline_recovery["recovery_hist_baseline"].iloc[-1]
    throughput_baseline = baseline_tph.iloc[-1]
    opt_increment = data_upstream_optimized["incremental_gold_produced"].iloc[-1]

    baseline_increment = (
        actual_head_gold_grade
        * recovery_baseline
        * throughput_baseline
        / params["oz_to_g"]
    )

    increment_uplift = opt_increment / baseline_increment - 1

    df_incremental_gold_produced = pd.DataFrame(
        {
            "actual_increment": [actual_increment],
            "optimized_increment": [opt_increment],
            "baseline_increment": [baseline_increment],
            "increment_uplift": [increment_uplift],
        }
    )
    return df_incremental_gold_produced
