import logging
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from optimizer.constraint import Penalty, Repair, repair
from optimizer.solvers import Solver
from optimizer.types import Matrix, Vector
from optimus_core.tag_dict import TagDict
from recommend.utils import get_on_features, get_possible_values

logger = logging.getLogger(__name__)

RepairList = List[Union[Repair, Callable[[Matrix], Matrix]]]
PenaltyList = List[Union[Penalty, Callable[[Matrix], Vector]]]

from recommend.domain import BaseDomainGenerator
from recommend.utils import (
    DuplicateIndexError,
    StatefulOptimizationProblem,
    _check_control_config,
    get_set_repairs,
    load_obj,
)


def create_models_dict(**models_dict) -> Dict:
    return models_dict


def make_problem(
    data: pd.DataFrame,
    td: TagDict,
    objective: Callable[[pd.DataFrame], pd.Series],
    domain_generator: Union[str, Type[BaseDomainGenerator]],
    repairs: Optional[Union[Repair, RepairList]] = None,
    penalties: Optional[Union[Penalty, PenaltyList]] = None,
    sense: Literal["minimize", "maximize"] = "maximize",
    areas_to_optimize: List = None,
) -> Tuple[Dict, Dict, List]:
    """Generate a problem dictionary for optimizer.

    Args:
        data: input data.
        td: tag dictionary.
        objective: objective function.
        domain_generator: domain generator class or import path to that class.
        repairs: a repair function or a list of repairs. Defaults to None.
        penalties: a penalty function or a list of
         penalty functions. Defaults to None.
        sense: either "maximize" or "minimize". Defaults to "maximize".
        areas_to_optimize: areas to optimize as per specified in params and picked from td

    Raises:
        DuplicateIndex: if input data contains duplicated index.

    Returns:
        A tuple of problem_dict and bounds_dict.
        problem_dict is a mapping of data.index to StatefulOptimizationProblem.
        domain_dict is a mapping of data.index to domain space.
    """
    if isinstance(domain_generator, str):
        domain_generator = load_obj(domain_generator)

    controls = td.select(
        condition=(
            lambda row: (
                row["tag_type"] == "control" and row["area"] in areas_to_optimize
            )
        )
    )

    _check_control_config(td, controls)

    # do not use parallel processing in the model
    # since we are parallelizing over rows in the dataframe
    if getattr(objective, "estimator__n_jobs", None) is not None:
        objective.set_params(estimator__n_jobs=1)

    problem_dict = {}
    domain_dict = {}
    off_idx_list = []

    # make `repairs` into a list if single repair is given
    if isinstance(repairs, Repair) or callable(repairs):
        repairs = [repairs]

    repairs = repairs or []

    if np.any(data.index.duplicated()):
        raise DuplicateIndexError("'data' must have unique index.")

    for idx in data.index:
        row = data.loc[[idx]]
        on_controls = get_on_features(row, td, controls)
        if not on_controls:
            off_idx_list.append(idx)
            continue

        repairs = repairs + get_set_repairs(td, on_controls)

        domain_space = domain_generator(
            td=td,
            data=row,
            optimizables=on_controls,
        ).generate()

        problem = StatefulOptimizationProblem(
            objective,
            state=row,
            optimizable_columns=on_controls,
            repairs=repairs,
            penalties=penalties,
            sense=sense,
        )
        problem_dict[idx] = problem
        domain_dict[idx] = domain_space

    if off_idx_list:
        off_idx_count = len(off_idx_list)
        logger.info(
            (
                f"{off_idx_count} rows are excluded since there is"
                " no active control variables."
            ),
        )

    return problem_dict, domain_dict, off_idx_list


def prepare_runs(
    scores: Dict[pd.Index, pd.DataFrame],
    iso_format: str = "%Y-%m-%dT%H:%M:%SZ",  # noqa: WPS323
    timestamp_col: str = "timestamp",
) -> List[Dict[str, str]]:
    """Create a list of runs in the format of `runs` endpoint of cra_api.

    Args:
        scores: a dictionary of optimization results.
        iso_format: format for timestamp.
        timestamp_col: column name for timestamp.

    Returns:
        An input to 'runs' endpoint of cra_api.
    """
    runs_for_cra_api = []
    for score_row in scores.values():
        score_row_timestamp = score_row[timestamp_col]
        score_row_timestamp = pd.to_datetime(score_row_timestamp)
        score_row[timestamp_col] = score_row_timestamp.dt.strftime(iso_format)

        timestamp = score_row.loc["curr", timestamp_col]
        run_id = score_row.loc["curr", "run_id"]
        runs_for_cra_api.append({"id": run_id, "data_timestamp": timestamp})
    return runs_for_cra_api


def add_model_predictions(
    models_dict: Dict, optim_results: Dict[int, pd.DataFrame]
) -> Dict[int, pd.DataFrame]:
    """Add curr/opt predicted values in ´optim_results´ for all models.

    Args:
        models_dict: dictionary of models
        optim_results: dict of dataframes containing optimization results

    Returns:
        Dataframe with optimization results and model predictions.
    """

    for scores in optim_results.values():
        for name, model in models_dict.items():
            scores[name] = model.predict(scores)

    return optim_results


def adjust_bounds_dict(
    data: pd.DataFrame,
    td: TagDict,
    bounds_dict: Dict,
    areas_to_optimize: Dict,
    bounds_update: List,
) -> Dict:
    """Adjust bounds dictionary for special cases"""
    # Case: only recommend to maintain or increase throughput
    tag_tph = bounds_update[0]
    op_max = td[tag_tph]["op_max"]
    max_delta = td[tag_tph]["max_delta"]

    controls = td.select(
        condition=(
            lambda row: (
                row["tag_type"] == "control" and row["area"] in areas_to_optimize
            )
        )
    )

    for idx in data.index:
        row = data.loc[[idx]]
        curr_tph_val = row[tag_tph].values[0]
        on_controls = get_on_features(row, td, controls)
        # Get throughput index and overwrite opt interval
        try:  # Throughput might not be a control variable
            idx_tph = on_controls.index(tag_tph)
            bounds_dict[idx][idx_tph] = (
                curr_tph_val,
                min(curr_tph_val + max_delta, max(op_max, curr_tph_val)),
            )
        except KeyError:  # If throughput is not control, skip
            continue
    return bounds_dict


def prepare_upstream_recommendations(
    scores: Dict[pd.Index, pd.DataFrame],
    td: TagDict,
    solver_dict: Dict[pd.Index, Solver],
    default_status: str = "Pending",
    on_control_only: bool = True,
    areas_to_optimize: List = None,
    translation_layer_tags: List = None,
    control_tags_not_for_ui: List = None,
) -> List[Dict[str, Any]]:
    """Creates a list of recommndations in the format of the `recommendations` endpoint.

    Args:
        scores: a dictionary of optimization results.
        td: tag dictionary.
        solver_dict: dictionary containing `Solver`.
        default_status: default status. Defaults to "Pending".
        on_control_only: if True, only export controls that are on. Defaults to True
        areas_to_optimize: areas to optimize as per specified in params and picked from td
        translation_layer_tags: Translation layer tags to output from recommendations on ui
        control_tags_not_for_ui: Control tags in tag dict that we do not want to display on UI
    Returns:
        An input to 'recommendations' endpoint of cra_api.
    """
    # Select specific controls as defined in areas to optimize parameters
    controls = td.select(
        condition=(
            lambda row: (
                row["tag_type"] == "control" and row["area"] in areas_to_optimize
            )
        )
    )
    if translation_layer_tags is None:
        final_control_list = list(set(controls) - set(control_tags_not_for_ui))
    else:
        final_control_list = list(
            set(controls + translation_layer_tags) - set(control_tags_not_for_ui)
        )

    recommendation_for_cra_api = []
    for score_row, solver in zip(scores.values(), solver_dict):
        on_controls = get_on_features(
            current_value=score_row.loc[["curr"]],
            td=td,
            controls=final_control_list,
        )
        controls_to_export = on_controls if on_control_only else final_control_list
        possible_values = get_possible_values(solver, on_controls)
        for control in controls_to_export:
            recommendation_for_cra_api.append(
                {
                    "tag": control,
                    "run_id": score_row["run_id"].values[0],
                    "recommended_value": score_row.loc["opt", control],
                    "id": str(uuid.uuid4()),
                    "recommendation_status": default_status,
                    "comment": "",
                    # "possible_values": possible_values.get(control, []),      # Uncomment only if you need it
                },
            )
    return recommendation_for_cra_api


def inject_upstream_opt(
    optim_results: Dict[int, pd.DataFrame], cil_recovery_input: pd.DataFrame
) -> pd.DataFrame:
    """Replace current values for mass pull and tailings sulphide grade, by optimized
    values coming from upstream optimization."""

    # Get optimized values for sulphide_grade
    sulphide_grade_dict = {
        optim_results[k]
        .loc["opt", "timestamp"]: optim_results[k]
        .loc["opt", "sulphide_grade"]
        for k in optim_results.keys()
    }
    # Get optimized values for mass_pull
    mass_pull_dict = {
        optim_results[k]
        .loc["opt", "timestamp"]: optim_results[k]
        .loc["opt", "mass_pull"]
        for k in optim_results.keys()
    }

    # Transform sulphide_grade dict to df
    sulphide_grade_df = pd.DataFrame.from_dict(
        sulphide_grade_dict, orient="index"
    ).rename(columns={0: "rougher_tails_sulphur_grade"})
    sulphide_grade_df.index.names = ["timestamp"]

    # Transform mass_pull dict to df
    mass_pull_df = pd.DataFrame.from_dict(mass_pull_dict, orient="index").rename(
        columns={0: "mass_pull"}
    )
    mass_pull_df.index.names = ["timestamp"]

    # Remove actual values for sulphide_grade and mass_pull from CIL input table
    cil_recovery_input.drop(
        columns=["rougher_tails_sulphur_grade", "mass_pull"], inplace=True
    )
    # Bring in the optimized values for sulphide_grade and mass_pull
    cil_recovery_input = cil_recovery_input.merge(
        sulphide_grade_df, how="left", on="timestamp"
    )
    cil_recovery_input = cil_recovery_input.merge(
        mass_pull_df, how="left", on="timestamp"
    )

    return cil_recovery_input


def get_baseline_tph(
    optim_results: Dict,
    baseline_tph_model: Dict,
    baseline_per_cluster: pd.DataFrame,
    baseline_features_median: pd.DataFrame,
) -> pd.DataFrame:
    """
    This function adjusts the final dataframes to be stored in db and to be used on UI for recommendations display
    Args:
        optim_results: optimization results dict from upstream
        baseline_tph_model: baseline tph cluster model based on historical data
        baseline_per_cluster: baseline median values of tph based on historical data
        baseline_features_median: Median of features in training data

    Returns: tph baseline
    """

    baseline_tph_features = [
        "sag_specific_power",
        "head_gold_grade",
        "head_sulphide",
        "tonnes_since_reline",
    ]

    baseline_tph_data = optim_results[0][baseline_tph_features][
        optim_results[0].index == "curr"
    ]

    if baseline_tph_data.isna().sum().sum() > 0:
        for x in baseline_tph_features:
            if baseline_tph_data[x].isna().sum() > 0:
                baseline_tph_data[x] = float(
                    baseline_features_median[
                        baseline_features_median["baseline_features"] == x
                    ]["baseline_features_median"]
                )
    scaler = MinMaxScaler()
    baseline_tph_data_scaled = scaler.fit(baseline_tph_data).transform(
        baseline_tph_data
    )
    y_kmeans = baseline_tph_model.predict(baseline_tph_data_scaled)

    baseline_tph = baseline_per_cluster[baseline_per_cluster.index == y_kmeans[0]][
        "baseline"
    ]

    return baseline_tph


def translation_layer(scores: Dict, params: Dict):
    """
    Translation layer to convert current features going in model (to recommend) with the ones that makes sense for
    operators to implement on
    Args:
        scores: recommend_results dictionary with timestamp and current and optimized values

    Returns: recommend_results dictionary with translation layer outputs

    """

    for key, value in scores.items():
        # froth velocity
        value[params["400_fc_004_froth_velocity"]] = np.where(
            value.index == "opt",
            value[params["flotation_cell_froth_velocity_diff_4_1"]]
            + value[params["400_fc_001_froth_velocity"]],
            value[params["400_fc_004_froth_velocity"]],
        )
        value[params["400_fc_002_froth_velocity"]] = np.where(
            value.index == "opt",
            value[params["flotation_cell_froth_velocity_diff_2_1"]]
            + value[params["400_fc_001_froth_velocity"]],
            value[params["400_fc_002_froth_velocity"]],
        )
        value[params["400_fc_003_froth_velocity"]] = np.where(
            value.index == "opt",
            (value[params["flotation_cell_froth_velocity_mean"]] * 4)
            - value[params["400_fc_001_froth_velocity"]]
            - value[params["400_fc_002_froth_velocity"]]
            - value[params["400_fc_004_froth_velocity"]],
            value[params["400_fc_003_froth_velocity"]],
        )

        # # air addition
        value[params["400_fc_004_air_addition"]] = np.where(
            value.index == "opt",
            value[params["flotation_cell_air_diff_4_1"]]
            + value[params["400_fc_001_air_addition"]],
            value[params["400_fc_004_air_addition"]],
        )
        value[params["400_fc_002_air_addition"]] = np.where(
            value.index == "opt",
            value[params["flotation_cell_air_diff_2_1"]]
            + value[params["400_fc_001_air_addition"]],
            value[params["400_fc_002_air_addition"]],
        )
        value[params["400_fc_003_air_addition"]] = np.where(
            value.index == "opt",
            (value[params["flotation_cell_air_mean"]] * 4)
            - value[params["400_fc_001_air_addition"]]
            - value[params["400_fc_002_air_addition"]]
            - value[params["400_fc_004_air_addition"]],
            value[params["400_fc_003_air_addition"]],
        )

        # TODO: Remove when op max constraint of 510 is relaxed
        if value["200_cv_001_weightometer"]["curr"] > 510:
            value["200_cv_001_weightometer"]["opt"] = value["200_cv_001_weightometer"][
                "curr"
            ]

    return scores


def _get_upstream_cfa(
    cfa_upstream: pd.DataFrame, td: TagDict, areas_to_optimize: List, target_tags: List
):
    """
    Transforms counterfactuals from upstream to vertical format
    Args:
        cfa_upstream: counterfactuals from upstream model
        td (object): Tag dictionary
        areas_to_optimize: list containing area

    Returns:
    Conterfactuals from upstream in vertical format
    """

    areas_to_optimize = areas_to_optimize
    control_tags = td.select(
        condition=(
            lambda row: (row["final_recs"] == 1 and row["area"] in areas_to_optimize)
        )
    )
    total_tags = control_tags + ["timestamp"] + target_tags

    cfa_upstream = pd.concat(cfa_upstream.values(), ignore_index=False)
    curr_cf = cfa_upstream[cfa_upstream[total_tags].index == "curr"][total_tags]
    opt_cf = cfa_upstream[cfa_upstream[total_tags].index == "opt"][total_tags]
    curr_cf.set_index("timestamp", inplace=True)
    opt_cf.set_index("timestamp", inplace=True)
    curr_cf = curr_cf.add_suffix("_curr")
    opt_cf = opt_cf.add_suffix("_opt")
    cfa_data = pd.merge(
        curr_cf, opt_cf, how="left", left_index=True, right_index=True
    ).sort_index(axis=1)
    return cfa_data


def combine_cfa_results(
    cfa_upstream: pd.DataFrame,
    cfa_downstream: pd.DataFrame,
    td: TagDict,
    areas_to_optimize: List,
    target_tags: List,
):
    """
    Combines the upstream and downstream counterfactuals of different frequencies
    into one file
    Args:
        td (object): tag dictionary
        cfa_upstream: file containing the counter factuals for upstream
        cfa_downstream: file containing the counter factuals for downstream
        areas_to_optimize: list containing thosse areas for whose tags we want to show
        on the UI

    Returns:
    Returns a combined file with counterfactuals for upstream and downstream model
    """
    cfa_upstream = _get_upstream_cfa(cfa_upstream, td, areas_to_optimize, target_tags)
    cfa_upstream["timestamp"] = cfa_upstream.index
    cfa_upstream.reset_index(drop=True, inplace=True)
    cfa_downstream.index = cfa_downstream["timestamp"]
    cfa_downstream.reset_index(drop=True, inplace=True)
    cfa_upstream["timestamp"] = pd.to_datetime(cfa_upstream["timestamp"], utc=True)
    cfa_downstream["timestamp"] = pd.to_datetime(cfa_downstream["timestamp"], utc=True)
    cfa_combined = pd.merge(
        cfa_upstream, cfa_downstream, how="left", on="timestamp"
    ).sort_index(axis=1)

    # Put the timestamp as the first column
    column_second = cfa_combined.pop("timestamp")
    cfa_combined.insert(0, "timestamp", column_second)
    return cfa_combined
