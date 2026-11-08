"""
Optimization Criteria and Scoring Algorithms.

Contains functions for calculating optimization scores and figure of merit
for atomic level selection.
"""

import numpy as np
from typing import Dict, Any
from .types import OptimizationCriteria


def calculate_optimization_score(result: Dict, criteria: OptimizationCriteria) -> float:
    """
    Calculate optimization score for a level combination.

    Args:
        result: Level selection result
        criteria: Optimization criteria

    Returns:
        Score between 0 and 1 (higher is better)
    """

    score = 0.0
    metrics = result['optimization_metrics']

    # Frequency accuracy (0-0.4 points)
    freq_error_mhz = abs(metrics.get('rf_frequency_error_hz', 1e9)) / 1e6  # Convert to MHz
    freq_score = max(0, 1.0 - freq_error_mhz / criteria.max_detuning_tolerance)
    score += 0.4 * freq_score

    # Dipole strength (0-0.4 points)
    dipole = metrics.get('dipole_rf', 0)
    if dipole > criteria.max_dipole_strength:
        dipole_score = 1.0
    else:
        dipole_score = dipole / criteria.max_dipole_strength
    score += 0.4 * dipole_score

    # Physics preferences (0-0.2 points)
    physics_score = 0.0

    rydberg1 = result['levels']['rydberg1']
    rydberg2 = result['levels']['rydberg2']

    # Prefer electric dipole transitions
    if abs(rydberg2.l - rydberg1.l) == 1 and criteria.prioritize_electric_dipole:
        physics_score += 0.5

    if criteria.prefer_low_n and rydberg1.n < 60 and rydberg2.n < 70:
        physics_score += 0.5

    score += 0.2 * physics_score

    return score


def calculate_analytical_figure_of_merit(wavelength_diff: float,
                                        n: int, l: int, j: float) -> float:
    """
    Calculate figure of merit for coupling transitions.

    Combines wavelength accuracy with physics-based preferences.
    """
    # Base score: exponential decay with wavelength difference
    wavelength_score = np.exp(-wavelength_diff / 10.0)  # 10 nm scale

    # Physics-based bonuses
    physics_score = 1.0

    # Prefer S and D states for coupling (common in Rydberg EIT)
    if l in [0, 2]:
        physics_score *= 1.2

    # Prefer lower n for stability
    n_bonus = np.exp(-(n - 50)**2 / 100)  # Peak around n=50

    return wavelength_score * physics_score * n_bonus


def calculate_rf_figure_of_merit(freq_diff: float,
                                n1: int, n2: int, delta_l: int) -> float:
    """
    Calculate figure of merit for RF transitions.

    Prefers close frequency matches with Δl=±1 transitions.
    """
    # Base score: exponential decay with frequency difference
    freq_score = np.exp(-freq_diff / 1e9)  # 1 GHz scale

    # Prefer smaller Δn for stronger transitions
    delta_n = abs(n2 - n1)
    delta_score = np.exp(-delta_n / 3.0)  # Prefer Δn=1-3

    # Prefer electric dipole allowed transitions
    if delta_l in [1]:
        physics_score = 1.3
    else:
        physics_score = 0.7

    return freq_score * delta_score * physics_score


def validate_optimization_criteria(criteria: OptimizationCriteria) -> bool:
    """
    Validate optimization criteria for reasonableness.

    Args:
        criteria: Optimization criteria to validate

    Returns:
        True if criteria are valid
    """
    if criteria.max_dipole_strength <= 0:
        return False
    if criteria.max_detuning_tolerance <= 0:
        return False
    if criteria.max_dipole_strength > 1e-20:  # Unrealistically strong dipole
        return False
    return True
