########################################################################
#                       Model train nodes
########################################################################

import logging
from typing import Dict

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import SilhouetteVisualizer

from optimus_core.tag_dict import TagDict

logger = logging.getLogger(__name__)


def _drop_na_columns(data: pd.DataFrame, target: str, ratio=0.3) -> pd.DataFrame:
    """
    Drops columns with high number of missing values
    Args:
        data: Shift level data
        target: empty col name
        ratio: Ratio above which a col is dropped . Ratio calculated by
        no. of rows with missing by total rows

    Returns: Dataframe with cols having ratio of missing value greater
    than cut off ratio dropped

    """
    too_many_na_columns = data.columns[data.isna().sum() > ratio * len(data)]
    if target in too_many_na_columns:
        too_many_na_columns = too_many_na_columns.drop(target)
    data = data.drop(columns=too_many_na_columns)
    return data


def _get_clusters(data: pd.DataFrame, params: Dict):
    """
    Trains the data using Kmeans clustering
    Args:
        data: Data at shift level
        params: parameters used in fitting the model

    Returns: Data at shift level along with scaled data at shift level
    and the model object trained on the data

    """
    init = params["train_params_tph"]["k_means_init"]
    n_init = params["train_params_tph"]["n_init"]
    max_iter = params["train_params_tph"]["max_iter"]
    random_state = params["train_params_tph"]["random_state"]
    n_init_best = params["train_params_tph"]["n_init_best"]
    clusters = params["train_params_tph"]["list_of_clusters"]

    scaler = MinMaxScaler()
    data_tph_cluster = data
    data_tph_cluster_scaled = scaler.fit(data_tph_cluster).transform(data_tph_cluster)
    best_k = None
    best_score = -1
    for i in clusters:
        """
        Create KMeans instances for different number of clusters
        """
        km = KMeans(
            n_clusters=i,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        q, mod = divmod(i - 3, 2)
        """
        Create SilhouetteVisualizer instance with KMeans instance
        Fit the visualizer
        """
        visualizer = SilhouetteVisualizer(km)
        visualizer.fit(data_tph_cluster_scaled)
        if visualizer.silhouette_score_ > best_score:
            best_score = visualizer.silhouette_score_
            best_k = i

    kmeans = KMeans(
        n_clusters=best_k,
        n_init=n_init_best,
        max_iter=max_iter,
        random_state=random_state,
    )
    kmeans.fit(data_tph_cluster_scaled)
    return data_tph_cluster, data_tph_cluster_scaled, kmeans


def baseline_tph_model(
    data: pd.DataFrame,
    td: TagDict,
    params: Dict,
):
    """
    Predict the cluster using ore characteristics for each
    timestamp
    Args:
        data: data
        td: tag dictionary
        params: parameter dictionary containing tag that has min
        throughput and cols used for first level clustering

    Returns: cluster model for baseline and baseline throughput median values for each cluster

    """
    throughput = params["train_params_tph"]["throughput"]
    data_tph_cluster = data.copy()
    features_for_cluster = params["train_params_tph"]["td_features_column"]
    data_tph_cluster = data_tph_cluster[td.select(features_for_cluster)]
    data_tph_cluster = data_tph_cluster.fillna(data_tph_cluster.median())
    (
        data_tph_cluster,
        data_tph_cluster_scaled,
        kmeans,
    ) = _get_clusters(data_tph_cluster, params)
    y_kmeans = kmeans.predict(data_tph_cluster_scaled)
    data["cluster_tph"] = y_kmeans
    baseline_per_cluster = (
        data.groupby("cluster_tph")[throughput].mean(numeric_only=True).to_frame()
    )
    baseline_features_median = data[data_tph_cluster.columns.to_list()].median()
    baseline_features_median = pd.DataFrame(
        {
            "baseline_features": baseline_features_median.index,
            "baseline_features_median": baseline_features_median.values,
        }
    )

    baseline_per_cluster.rename(
        columns={"200_cv_001_weightometer": "baseline"}, inplace=True
    )
    return baseline_per_cluster, kmeans, baseline_features_median


def _get_df_clusters_tph(df):
    """
    Reformat data of throughput baseline to match the format of common baseline table
    Args:
        df: data containing baseline and clusters for throughput

    Returns:
    baseline data with clusters for throughput
    """
    df["tag_id"] = "tph"
    df["cluster"] = df.index
    df["operating_modes"] = "None"
    df = df.reset_index(drop=True)
    df = df.reindex(
        columns=[col for col in df.columns if col != "baseline"] + ["baseline"]
    )
    return df


def _get_df_all_clusters_and_opms_gold_recovery(df):
    """
    Reformat data of gold_recovery baseline to match the format of common baseline table
    Args:
        df: data containing baseline, clusters and opm for gold_recovery

    Returns:
    baseline data with clusters and opm for gold_recovery
    """
    df_comb = pd.DataFrame()
    for i, j in df.items():
        df1 = j
        df1["cluster"] = i
        df1["operating_modes"] = df1.index.astype(int)
        df1 = df1.reset_index(drop=True)
        df_comb = pd.concat([df_comb, df1], ignore_index=True)
    df_comb["tag_id"] = "gold_recovery"
    df_comb["baseline"] = df_comb["mean"]
    df_final = df_comb[["tag_id", "cluster", "operating_modes", "baseline"]]
    df_final = pd.DataFrame(df_final)
    return df_final


def baseline_historic_upstream_and_downstream(
    baseline_tph: pd.DataFrame, baseline_cil_recovery: pd.DataFrame, td: TagDict
):
    """
    Combine data of throughput and gold_recovery baselines
    Args:
        baseline_tph: data containing baseline and clusters for throughput
        baseline_cil_recovery: data containing baseline, clusters and opm for gold_recovery
        td: tag dictionary

    Returns:
    Combined data of throughput and gold_recovery baselines
    """
    baseline_tph = _get_df_clusters_tph(baseline_tph)
    baseline_cil_recovery = _get_df_all_clusters_and_opms_gold_recovery(
        baseline_cil_recovery
    )
    baseline_historic_upstream_and_downstream = pd.concat(
        [baseline_tph, baseline_cil_recovery], ignore_index=False
    )
    return baseline_historic_upstream_and_downstream


def get_median_for_baseline_recovery(data: pd.DataFrame, td: TagDict, params: Dict):
    """
    Calculate the median value of each variable used in downstream model from train data
    Args:
        data: shift level train data
        td: tag dictionary
        params: Dictionary containing parameters

    Returns:
    Returns the median value of each variable used in downstream model from train data
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
    df_leaching = data[leaching_vars].median()

    return df_leaching
