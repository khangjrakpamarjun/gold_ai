########################################################################
#                       Model train nodes
########################################################################

import logging
from typing import Dict

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import SilhouetteVisualizer

logger = logging.getLogger(__name__)


def _check_null_threshold_in_controls(
    data: pd.DataFrame, target: str, ratio
) -> pd.DataFrame:
    """
    Checks if control tags have nulls more than the cutoff and raises value error message if so
    Args:
        data: Shift level data
        target: cols to be ignored from checking for missing value cutoff
        ratio: Ratio above which a col is dropped . Ratio calculated by
        no. of rows with missing by total rows

    Returns:
        Dataframe with controls having ratio of missing value lesser than cut off ratio

    """
    too_many_na_columns = data.columns[data.isna().sum() > ratio * len(data)]
    if target in too_many_na_columns:
        too_many_na_columns = too_many_na_columns.drop(target)
    if len(too_many_na_columns) != 0:
        raise ValueError(
            f"Control variable/variables {too_many_na_columns} have more than {ratio*100:.0f} %  NA"
            " values overall, please check the quality of data"
        )
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
    init = params["train_params"]["k_means_init"]
    n_init = params["train_params"]["n_init"]
    max_iter = params["train_params"]["max_iter"]
    random_state = params["train_params"]["random_state"]
    n_init_best = params["train_params"]["n_init_best"]
    clusters = params["train_params"]["list_of_clusters"]

    scaler = MinMaxScaler()
    data_upstream_cluster = data
    data_upstream_cluster_scaled = scaler.fit(data_upstream_cluster).transform(
        data_upstream_cluster
    )
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
        """
        Create SilhouetteVisualizer instance with KMeans instance
        Fit the visualizer
        """
        visualizer = SilhouetteVisualizer(km)
        visualizer.fit(data_upstream_cluster_scaled)
        if visualizer.silhouette_score_ > best_score:
            best_score = visualizer.silhouette_score_
            best_k = i

    kmeans = KMeans(
        n_clusters=best_k,
        n_init=n_init_best,
        max_iter=max_iter,
        random_state=random_state,
    )
    kmeans.fit(data_upstream_cluster_scaled)
    return data_upstream_cluster, data_upstream_cluster_scaled, kmeans


def get_first_level_clusters(
    data: pd.DataFrame,
    td: pd.DataFrame,
    params: Dict,
):
    """
    Predict the first level cluster using ore characteristics for each
    timestamp
    Args:
        data: Shift level data
        td: tag dictionary
        params: parameter dictionary containing tag that has min
        throughput and cols used for first level clustering

    Returns: Shift level data  with only ore characteristics with
    clusters assigned to each timestamp

    """
    cluster_kmeans = params["cluster_col_name"]
    features_for_cluster = params["feature_downstream_cluster"]
    data_upstream_cluster = data[td.select(features_for_cluster)]
    data_upstream_cluster = data_upstream_cluster.fillna(data_upstream_cluster.median())

    (
        data_upstream_cluster,
        data_upstream_cluster_scaled,
        kmeans,
    ) = _get_clusters(data_upstream_cluster, params)
    y_kmeans = kmeans.predict(data_upstream_cluster_scaled)
    model_features_first_level_cluster = data_upstream_cluster.columns.to_list()
    data_upstream_cluster[cluster_kmeans] = y_kmeans
    return (
        data_upstream_cluster,
        data_upstream_cluster_scaled,
        data,
        kmeans,
        model_features_first_level_cluster,
    )


def get_operating_mode_per_cluster(
    td: pd.DataFrame,
    data_upstream_cluster: pd.DataFrame,
    df_shift: pd.DataFrame,
    params: Dict,
):
    """
    Predict the second level cluster or best operating  using process
    variables  for each timestamp

    Args:
        td: Tag dictionary
        data_upstream_cluster: Shift level data containing  ore
        characteristics tags ( 6 variables) and ore clusters
        df_shift: Shift level data containing all columns
        params: parameter dictionary containing cut off ratio of
        missing value above which col will be dropped

    Returns: Dataframe with first level clusters along with their
    optimal operating modes/clusters. Also returns a dictionary
    containing the mapping of each clusters with their best operating
    modes.

    """
    # TODO: Check where is recovery column created in pipeline and
    #  remove it and instead use only gold_recovery everywhere
    feature_operating_modes = params["feature_operating_modes"]
    cluster_kmeans = params["cluster_col_name"]
    recovery_column = params["recovery"]
    dictionary = td
    best_operating_mode_per_cluster = {}
    train_model_per_cluster = {}
    all_operating_mode_per_cluster = {}
    model_features_second_level_clusterwise = {}
    model_features_second_level_cluster = []
    control_variables = td.select(
        condition=(
            lambda row: (row["tag_type"] == "control" and row[feature_operating_modes])
        )
    )
    cil_data = df_shift[dictionary.select("feature_operating_modes")]

    # Filter only the control variables
    cil_data = cil_data[control_variables]

    # Drop shifts from  data with only ore clusters and variables where ore clusters are predicted as null
    data_upstream_cluster = data_upstream_cluster[
        data_upstream_cluster[cluster_kmeans].notna()
    ]

    # Check if control variables have more than 30 % NA values overall
    cil_data = _check_null_threshold_in_controls(
        cil_data, target="", ratio=params["remove_na_ratio"]
    )

    # Fill control variables that have less than 30 % missing values with their median values
    cil_data.fillna(cil_data.median(), inplace=True)

    # Join the ore clusters column to the data containing control variables
    cil_data_selected = cil_data.join(data_upstream_cluster[[cluster_kmeans]])
    cil_data_selected.dropna(subset=[cluster_kmeans], inplace=True)
    cil_data_selected[cluster_kmeans] = cil_data_selected[cluster_kmeans].astype(int)
    cil_data_shift = cil_data_selected.copy()
    data_process_combined = pd.DataFrame()
    df_shift_copy = df_shift
    unique_cluster_opm = data_upstream_cluster[cluster_kmeans].unique()
    for k in range(len(unique_cluster_opm)):
        cluster = unique_cluster_opm[k]
        data_process = cil_data_shift[cil_data_shift[cluster_kmeans] == cluster].drop(
            "cluster_kmeans", axis=1
        )
        model_features_second_level_clusterwise[
            str(cluster)
        ] = data_process.columns.to_list()
        model_features_second_level_cluster.extend(
            model_features_second_level_clusterwise[str(cluster)]
        )

        # TODO: check if saving features for each cluster is required
        # Removing duplicate features
        model_features_second_level_cluster = list(
            set(model_features_second_level_cluster)
        )
        data_process, data_process_scaled, kmeans = _get_clusters(data_process, params)
        operating_modes = kmeans.predict(data_process_scaled)
        data_process[f"operating_modes_{cluster}"] = operating_modes
        df_shift_copy = df_shift_copy.join(data_process[[f"operating_modes_{cluster}"]])
        all_operating_mode_per_cluster[int(cluster)] = df_shift_copy.groupby(
            f"operating_modes_{cluster}"
        )[recovery_column].describe()
        best_index = (
            pd.DataFrame(
                df_shift_copy.groupby(f"operating_modes_{cluster}").median(
                    numeric_only=True
                )
            )[[recovery_column]]
            .mean(axis=1)
            .idxmax()
        )
        best_operating_mode_per_cluster[str(cluster)] = int(best_index)
        train_model_per_cluster[str(cluster)] = kmeans
        data_process_combined = pd.concat([data_process_combined, data_process], axis=0)
    df_shift_combined = df_shift_copy.join(cil_data_shift[[cluster_kmeans]])
    recoveries_per_cluster = pd.DataFrame()
    columns_export = (
        list(cil_data.columns)
        + ["mass_pull", recovery_column]
        + list(data_upstream_cluster.columns)
        + [
            f"operating_modes_{cluster}"
            for cluster in range(len(data_upstream_cluster[cluster_kmeans].unique()))
        ]
    )

    data = df_shift_combined[columns_export]
    for column in data:
        data[column] = data[column].mask(
            (data[column] < data[column].quantile(0.02))
            | (data[column] > data[column].quantile(0.98))
        )
    # Put all the operating modes of each clusters under one column called
    # operating_modes
    operating_modes_per_cluster = [
        f"operating_modes_{cluster}"
        for cluster in range(
            len([x for x in data[cluster_kmeans].unique().tolist() if str(x) != "nan"])
        )
    ]
    data["operating_modes"] = data[operating_modes_per_cluster].sum(axis=1, min_count=1)
    df_processed_with_clusters = data
    model_features = [
        i
        for i in data.columns.to_list()
        if i
        not in operating_modes_per_cluster
        + ["cluster_kmeans", "operating_modes", "mass_pull"]
    ]

    return (
        df_processed_with_clusters,
        best_operating_mode_per_cluster,
        train_model_per_cluster,
        recoveries_per_cluster,
        model_features,
        all_operating_mode_per_cluster,
        model_features_second_level_cluster,
        model_features_second_level_clusterwise,
        data_process_scaled,
    )
