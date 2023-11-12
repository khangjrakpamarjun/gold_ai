from typing import Dict

import numpy as np
import pandas as pd

from feature_factory import FeatureFactory


def prepare_custom_function_params_for_feature_factory(params):
    for feat in params:
        if not params[feat]["function"].startswith("feature_factory"):
            params[feat]["function"] = eval(params[feat]["function"])
    return params


def create_features(
    data: pd.DataFrame, params: Dict, user_inputs_form_conversion: pd.DataFrame
) -> pd.DataFrame:
    params = prepare_custom_function_params_for_feature_factory(
        params["feature_factory"]
    )
    transformer = FeatureFactory(params)
    data_with_features_intermediate = transformer.fit_transform(data)
    data_with_features = balls_weight_addition(
        data_with_features_intermediate, user_inputs_form_conversion
    )
    return data_with_features


#######################################################################################
#                              grinding custom functions
#######################################################################################


def loading_balls_to_sag(data, dependencies):
    return data[dependencies[0]] - data[dependencies[0]].shift(1)


def sag_mill_density(data, dependencies, alpha=1.0):
    return (
        alpha * data[dependencies[0]] / (data[dependencies[0]] + data[dependencies[1]])
    )


#######################################################################################
#                              total Recovery custom functions
#######################################################################################


def filter_column(data, dependencies):
    return data[dependencies[0]]


def regrind_density_solids(data, dependencies, pyrite_sg, density_solids_factor):
    """
    This function corrects regrind sulphide based on factor provided by client
    Args:
        data: 30 min data
        params: Derived feature parameters

    Returns:

    """
    sulphide_in_pyrite = 32.06 * 2 / 119.965 * 100

    regrind_sulphide_corrected = (
        2.2
        * (
            (pyrite_sg * data[dependencies[0]] / sulphide_in_pyrite)
            + 2.8 * (1 - data[dependencies[0]] / sulphide_in_pyrite)
        )
        / (
            3.2
            * (
                pyrite_sg * (data[dependencies[0]] / sulphide_in_pyrite)
                + 2.8 * (1 - data[dependencies[0]] / sulphide_in_pyrite)
                - 1
            )
        )
    )

    return density_solids_factor * regrind_sulphide_corrected


def regrind_density(data, dependencies, density_liquid_factor):
    return -(data[dependencies[1]] * density_liquid_factor) / (
        (data[dependencies[1]] - density_liquid_factor) * data[dependencies[0]] / 100
        - data[dependencies[1]]
    )


def regrind_mass_flow_ore(data, dependencies):
    return data[dependencies[0]] * data[dependencies[1]] / 100


def final_tails_density(
    data, dependencies, density_liquid_factor, density_solids_factor
):
    return -(density_solids_factor * density_liquid_factor) / (
        (density_solids_factor - density_liquid_factor) * data[dependencies[0]] / 100
        - density_solids_factor
    )


def final_tails_mass_flow_ore(data, dependencies):
    return data[dependencies[0]] * data[dependencies[1]] / 100


def weighted_grade_regrind(data, dependencies):
    return (data[dependencies[0]] * data[dependencies[1]]) / data[dependencies[2]]


def weighted_grade_rougher(data, dependencies):
    return (data[dependencies[0]] * data[dependencies[1]]) / data[dependencies[2]]


def head_sulphide(data, dependencies):
    return (
        data[dependencies[0]] * data[dependencies[1]]
        + data[dependencies[2]] * data[dependencies[3]]
    ) / data[dependencies[4]]


def head_grade(data, dependencies):
    return (
        data[dependencies[0]] * data[dependencies[1]]
        + data[dependencies[2]] * data[dependencies[3]]
    ) / data[dependencies[4]]


def weighted_grade_regrind_shifted(
    data, dependencies, regrind_retention_time, num_data_points_in_shift
):
    return data[dependencies[0]].shift(
        regrind_retention_time * num_data_points_in_shift
    )


def weighted_grade_rougher_shifted(
    data, dependencies, rougher_retention_time, num_data_points_in_shift
):
    return data[dependencies[0]].shift(
        rougher_retention_time * num_data_points_in_shift
    )


def head_sulphide_shifted(
    data, dependencies, rougher_retention_time, num_data_points_in_shift
):
    return data[dependencies[0]].shift(
        rougher_retention_time * num_data_points_in_shift
    )


def relative_diff(data, dependencies):
    return (data[dependencies[0]] - data[dependencies[1]]) / data[dependencies[0]]


def recovery_previous_shift(data, dependencies, num_data_points_in_shift):
    return data[dependencies[0]].shift(num_data_points_in_shift)


def mean_value(data, dependencies):
    return data[dependencies].mean(axis=1)


def units_conversion(df, dependencies, volume):
    return (df[dependencies[0]] / volume) * 1000


#######################################################################################
#                              Flotation Recovery custom functions
#######################################################################################


def flotation_recovery(data, dependencies):
    return (data[dependencies[0]] * data[dependencies[1]]) / (
        data[dependencies[2]] * data[dependencies[3]]
    )


def flotation_recovery(data, dependencies):
    return (data[dependencies[0]] * data[dependencies[1]]) / (
        data[dependencies[2]] * data[dependencies[3]]
    )


#######################################################################################
#                              Flotation model custom functions
#######################################################################################


def rougher_tails_sulphide_sulphur_rolled(data, dependencies, steps, delay):
    return (
        data[dependencies[0]]
        .rolling(window=steps)
        .mean()
        .shift(freq="1H", periods=delay)
    )


def mass_pull_prev(data, dependencies, lag_hours):
    df = data[[dependencies[0]]].copy()
    mass_pull_prev_df = data[[dependencies[0]]].copy()
    mass_pull_prev_df.index = mass_pull_prev_df.index + pd.Timedelta(hours=lag_hours)
    df = df.join(mass_pull_prev_df, rsuffix="_prev")
    return df["mass_pull_prev"]


def tonnes_since_reline(df, dependencies, sag_reline_dates):
    data = df.copy()

    data["sag_reline_dates"] = np.where(data.index.isin(sag_reline_dates), 1, np.nan)
    data = data.reset_index()
    indexes = data.index[data["sag_reline_dates"] == 1].tolist()
    l_mod = [0] + indexes + [data.shape[0]]
    list_of_dfs = [data.iloc[l_mod[n] : l_mod[n + 1]] for n in range(len(l_mod) - 1)]
    for i in list_of_dfs:
        i["tonnes_since_reline"] = i[dependencies[0]].cumsum(skipna=True)
        i["tonnes_since_reline"] = i["tonnes_since_reline"] / 2
    data = pd.concat(list_of_dfs)
    data.set_index("timestamp", inplace=True)
    return data["tonnes_since_reline"]


def balls_weight_addition(data, user_inputs_form_conversion):
    data["balls_weight_rounded"] = data["300_fe_001_level"].round()
    data.reset_index(inplace=True)
    data = data.merge(
        user_inputs_form_conversion,
        how="left",
        left_on="balls_weight_rounded",
        right_on="balls_weight_percentage",
    )
    data["balls_difference"] = data["balls_weight_kg"] - data["balls_weight_kg"].shift(
        -1
    )
    data["balls_weight_addition"] = np.where(
        data["balls_difference"] > 0, data["balls_difference"], np.nan
    )
    data.drop(
        ["balls_difference", "balls_weight_rounded", "balls_weight_kg"],
        axis=1,
        inplace=True,
    )
    data.set_index("timestamp", inplace=True)
    return data


def reagent_split(data, dependencies, coefficient, noise_reduction, time_conversion):
    return (
        (
            (data[dependencies[0]] * time_conversion)
            / (data[dependencies[2]] * noise_reduction)
        )
        * coefficient
    ) / data[dependencies[1]]


def increment(data, dependencies, oz_to_g):
    return (
        data[dependencies[0]] * data[dependencies[1]] * data[dependencies[2]]
    ) / oz_to_g
