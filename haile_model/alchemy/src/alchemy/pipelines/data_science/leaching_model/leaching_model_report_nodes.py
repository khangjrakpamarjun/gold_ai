##########################################################################################
#                       Model report nodes
##########################################################################################
import logging
import pickle
from typing import Dict

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def model_report_information(params: Dict):
    logger.info(params["model_report_name"])
    return None


def get_2d_ore_cluster_plot(
    params: Dict,
    td: pd.DataFrame,
    data_scaled,
    first_level_cluster_trained_model: pickle,
):
    """
    Generates the 2D ore cluster plot for train data for analysis purpose. Testing
    Args:
        params: dictionary containing parameters
        td: tag dictionary
        data_scaled: the train data that was scaled using a suitable scaler ( minmax scaler)
        first_level_cluster_trained_model: the saved train model object for ore cluster

    Returns:
    Returns the 2D ore cluster plot for train data
    """
    kmeans = first_level_cluster_trained_model
    pca = PCA(n_components=3)
    pca.fit(data_scaled)
    data_ore_pca = pca.transform(data_scaled)
    pca.explained_variance_ratio_.sum()

    # Get unique cluster labels
    unique_labels = np.unique(kmeans.labels_)

    # Assign colors to unique cluster labels
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    # 2D plot
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    for label, color in zip(unique_labels, colors):
        mask = kmeans.labels_ == label
        ax.scatter(
            data_ore_pca[mask, 0],
            data_ore_pca[mask, 1],
            color=color,
            label=f"Cluster {label}",
            s=10,
        )
    ax.set_xlabel("pca1")
    ax.set_ylabel("pca2")
    ax.set_title("2D Cluster Plot")
    ax.legend()
    return plt


def get_3d_ore_cluster_plot(
    params: Dict,
    td: pd.DataFrame,
    data_scaled,
    first_level_cluster_trained_model: pickle,
):
    """
    Generates the 3D ore cluster plot for train data
    Args:
        params: dictionary containing parameters
        td: tag dictionary
        data_scaled: the train data that was scaled using a suitable scaler ( minmax scaler)
        first_level_cluster_trained_model: the saved train model object for ore cluster

    Returns:
    Returns the 3D ore cluster plot for train data
    """
    kmeans = first_level_cluster_trained_model
    pca = PCA(n_components=3)
    pca.fit(data_scaled)
    data_ore_pca = pca.transform(data_scaled)
    pca.explained_variance_ratio_.sum()

    # Get unique cluster labels
    unique_labels = np.unique(kmeans.labels_)

    # Assign colors to unique cluster labels
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    # 3D plot
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    for label, color in zip(unique_labels, colors):
        mask = kmeans.labels_ == label
        ax.scatter(
            data_ore_pca[mask, 0],
            data_ore_pca[mask, 1],
            data_ore_pca[mask, 2],
            color=color,
            label=f"Cluster {label}",
            s=10,
        )
    ax.set_xlabel("pca1")
    ax.set_ylabel("pca2")
    ax.set_zlabel("pca3")
    ax.set_title("3D Cluster Plot")
    ax.legend()
    return plt


def _get_ore_cluster_profile(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generates the cluster profile table for all tags in the data
    Args:
        data: shift level data

    Returns:
    Returns the cluster profile table for all tags in the data with
    summary statistics for all tags
    """
    # TODO: Add operating modes as a column for each tags
    cluster = data["cluster_kmeans"].unique().tolist()
    combine_df = pd.DataFrame()
    for i in range(len(cluster)):
        cls = cluster[i]
        cluster_wise = data[data["cluster_kmeans"] == cluster[cls]]
        operating_modes = cluster_wise["operating_modes"].unique().tolist()
        for j in range(len(operating_modes)):
            opm = int(operating_modes[j])
            cluster_wise_data = cluster_wise[
                cluster_wise["operating_modes"] == operating_modes[opm]
            ]
            first_df = cluster_wise_data.describe()
            first_df = pd.DataFrame(first_df.T)
            first_df = first_df.reset_index()
            first_df["cluster"] = cls
            first_df["operating_modes"] = opm

            # stack the two DataFrames
            combine_df = pd.concat([combine_df, first_df], axis=0)
    return combine_df


def _get_ore_cluster_profile_cfa(data: pd.DataFrame) -> pd.DataFrame:
    """

    Generates the cluster profile table for all tags for the data
    Args:
        data: shift level data used in cfa

    Returns:
    Returns the cluster profile table for all tags for the data used in cfa
    with summary statistics for all tags
    """
    cluster1 = data["ore_cluster_predicted"].unique().tolist()
    combine_df = pd.DataFrame()
    for i in cluster1:
        cls = i
        cluster_wise = data[data["ore_cluster_predicted"] == i]
        first_df = cluster_wise.describe()
        first_df = pd.DataFrame(first_df.T)
        first_df = first_df.reset_index()
        first_df["ore_cluster_predicted"] = cls
        # stack the two DataFrames
        combine_df = pd.concat([combine_df, first_df], axis=0)
    return combine_df


def get_ore_cluster_profile_for_train_data(
    train_data: pd.DataFrame, params: Dict, td: pd.DataFrame
) -> pd.DataFrame:
    """
    Calls the function to generates the cluster profile table for all
    tags in the train data
    Args:
        train_data: shift level train data
        params: dictionary containing parameters
        td: tag dictionary

    Returns:
    Returns the cluster profile table for all tags in the train data
    """
    ore_cluster_profile_for_train_data = _get_ore_cluster_profile(train_data)
    return ore_cluster_profile_for_train_data


def get_ore_cluster_profile_for_cfa(
    train_data: pd.DataFrame, params: Dict, td: pd.DataFrame
) -> pd.DataFrame:
    """
    Calls the function to generates the cluster profile table for all
     tags for the data used in cfa
    Args:
        train_data: shift level data used in cfa
        params: dictionary containing parameters
        td: tag dictionary

    Returns:
    Returns the cluster profile table for all tags for the data used in cfa
    """
    ore_cluster_profile_for_train_data = _get_ore_cluster_profile_cfa(train_data)
    return ore_cluster_profile_for_train_data


def get_box_plots(train_data: pd.DataFrame, params: Dict, td: pd.DataFrame):
    """
    Generates box plots for the ore characteristics(6x) and recovery tags
    Args:
        train_data: shift level data
        params: dictionary containing parameters
        td: tag dictionary

    Returns:
    Returns box plots for the ore characteristics(6x) and recovery tags
    """
    col = [
        "regrind_product_gold_grade",
        "regrind_product_sulphide_sulphur",
        "rougher_tails_gold_grade",
        "rougher_tails_sulphur_grade",
        "regrind_product_p80",
        "primary_cyclone_overflow_p80",
        "recovery",
    ]

    cluster = train_data["cluster_kmeans"].unique().tolist()
    # increase the figsize for better resolution of subplots
    fig, axs = plt.subplots(len(col), 1, figsize=(12, 8 * len(col)))

    for j, ax in enumerate(axs):
        ore_char = col[j]
        boxplot_data = []
        labels = []
        colors = []

        for i, cluster_val in enumerate(cluster):
            cluster_data = train_data[train_data["cluster_kmeans"] == cluster_val]
            non_null_data = cluster_data[ore_char].dropna()
            if len(non_null_data) > 0:
                boxplot_data.append(non_null_data)
                labels.append(str(cluster_val))
                colors.append(cm.tab10(i))  # Using tab10 colormap for distinct colors

        if len(boxplot_data) > 0:
            boxplot = ax.boxplot(boxplot_data, patch_artist=True, labels=labels)

            # Assign colors to box plots
            for box, color in zip(boxplot["boxes"], colors):
                box.set(facecolor=color)

            ax.set_xlabel("cluster_kmeans")
            ax.set_ylabel(ore_char)
            ax.set_title("feature is " + str(ore_char))

    # Adjust spacing between subplots
    plt.tight_layout()

    return plt


def get_box_plots_cfa(train_data: pd.DataFrame, params: Dict, td: pd.DataFrame):
    """
    Generates box plots for the ore characteristics(6x) and recovery tags
    for the data used in cfa
    Args:
        train_data: shift level data used in cfa
        params: dictionary containing parameters
        td: tag dictionary

    Returns:
    Returns box plots for the ore characteristics(6x) and recovery tags
    for the data used in cfa
    """
    col = [
        "regrind_product_gold_grade",
        "regrind_product_sulphide_sulphur",
        "rougher_tails_gold_grade",
        "rougher_tails_sulphur_grade",
        "regrind_product_p80",
        "primary_cyclone_overflow_p80",
        "recovery_actual",
        "recovery_optimized",
        "recovery_predicted",
    ]

    cluster = train_data["ore_cluster_predicted"].unique().tolist()
    # increase the figsize for better resolution of subplots
    fig, axs = plt.subplots(len(col), 1, figsize=(12, 8 * len(col)))

    for j, ax in enumerate(axs):
        ore_char = col[j]
        boxplot_data = []
        labels = []
        colors = []

        for i, cluster_val in enumerate(cluster):
            cluster_data = train_data[
                train_data["ore_cluster_predicted"] == cluster_val
            ]
            non_null_data = cluster_data[ore_char].dropna()
            if len(non_null_data) > 0:
                boxplot_data.append(non_null_data)
                labels.append(str(cluster_val))
                colors.append(cm.tab10(i))  # Using tab10 colormap for distinct colors

        if len(boxplot_data) > 0:
            boxplot = ax.boxplot(boxplot_data, patch_artist=True, labels=labels)

            # Assign colors to box plots
            for box, color in zip(boxplot["boxes"], colors):
                box.set(facecolor=color)

            ax.set_xlabel("ore_cluster_predicted")
            ax.set_ylabel(ore_char)
            ax.set_title("feature is " + str(ore_char))

    # Adjust spacing between subplots
    plt.tight_layout()

    return plt
