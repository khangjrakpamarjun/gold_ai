# Copyright (c) 2016 - present
# QuantumBlack Visual Analytics Ltd (a McKinsey company).
# All rights reserved.
#
# This software framework contains the confidential and proprietary information
# of QuantumBlack, its affiliates, and its licensors. Your use of these
# materials is governed by the terms of the Agreement between your organisation
# and QuantumBlack, and any unauthorised use is forbidden. Except as otherwise
# stated in the Agreement, this software framework is for your internal use
# only and may only be shared outside your organisation with the prior written
# permission of QuantumBlack.
from typing import Tuple

import numpy as np


def perturb_data(solutions: np.ndarray, perturbations: np.ndarray) -> Tuple[np.ndarray]:
    """
    Perturb the 2d array of solutions. Each solution will be locally perturbed
    to check for robustness, perturbed by "noise"

    Args:
        solutions: 2d np.ndarray of potential optimization solutions
        perturbations: 3d np.ndarray of perturbations/noise to apply to each sol.

    Returns:
        Tuple of 2 arrays: the "flattened" array of perturbations, and the
         multi-dimensional array of perturbed data
    """
    perturbed_df = solutions[..., np.newaxis] + perturbations

    # Re-order axes/transpose
    dft = np.transpose(perturbed_df, (0, 2, 1))
    # flatten to have same columns as solutions
    flattened_transposed = dft.reshape(-1, solutions.shape[1])
    return flattened_transposed, perturbed_df


def process_objectives(
    perturbed_data: np.ndarray, objectives: np.ndarray, n_perturb: float
) -> Tuple[np.ndarray]:
    """
    Given an array of perturbed solutions, let's extract the ones corresponding
    to the largest solution for each candidate particle.

    Args:
        perturbed_data: 3d nd-array of perturbed data
        objectives: 1d np.array of objective values for all candidate perturbations
        n_perturb: Number of perturbations each candidate experienced

    Returns:
        Tuple: The solutions corresponding to the "worst" solutions after applying
        perturbations, and the corresponding "worst" objectives.
    """
    mapped_obj = objectives.reshape(-1, n_perturb)
    # find the vector of "worst" objective indices
    max_id = np.argmax(mapped_obj, axis=1)
    # index into the tensor/array of perturbed indices to find the nearest/best one
    return perturbed_data[np.arange(len(max_id)), :, max_id], np.max(mapped_obj, axis=1)
