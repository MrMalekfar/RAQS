"""
Calculate all three detunings from optimized atomic levels and laser/RF parameters.
"""

import logging
import numpy as np
from typing import Tuple, Dict
from .calculate_probe_detuning import calculate_probe_detuning
from .calculate_coupling_detuning import calculate_coupling_detuning
from .calculate_rf_detuning import calculate_rf_detuning
from ...config.laser_rf_config import LASER_PARAMS, RF_PARAMS
from ...exceptions import PhysicsCalculationError, PhysicsConfigurationError

logger = logging.getLogger(__name__)


def calculate_all_detunings(atom, levels: Dict, probe_params: Dict, coupling_params: Dict, rf_params: Dict) -> Tuple[float, float, float]:
    """
    Calculate all three detunings from optimized atomic levels and laser/RF parameters.
    Imports laser wavelengths and RF frequency from laser_rf_config.py to avoid hardcoded values.

    Args:
        atom: ARC atom object
        levels: Dictionary with 'ground', 'excited', 'rydberg1', 'rydberg2' levels
        laser_params: Dictionary with probe and coupling laser parameters
        rf_params: Dictionary with RF field parameters

    Returns:
        tuple: (delta_p_rad_us, delta_c_rad_us, delta_rf_rad_us)
    """
    logger.info("=== Calculating Systematic Detunings ===")
    print(f"DEBUG: probe_params = {probe_params}")
    print(f"DEBUG: 'wavelength_nm' in probe_params: {'wavelength_nm' in probe_params}")

    # Use passed parameters (not globals) to get current wavelengths/frequencies
    try:
        probe_wavelength = probe_params['wavelength_nm']
        coupling_wavelength = coupling_params['wavelength_nm']
        rf_frequency = rf_params['frequency_hz']
    except KeyError as exc:
        # F05: a missing wavelength/frequency key must never be replaced by a
        # default value.
        raise PhysicsConfigurationError(
            f"Missing required detuning input {exc} - probe_params keys="
            f"{sorted(probe_params)}, coupling_params keys="
            f"{sorted(coupling_params)}, rf_params keys={sorted(rf_params)}. "
            "F05: no fallback is permitted."
        ) from exc

    for name, value in (('probe_wavelength_nm', probe_wavelength),
                        ('coupling_wavelength_nm', coupling_wavelength),
                        ('rf_frequency_hz', rf_frequency)):
        if value is None or not np.isfinite(value) or value <= 0:
            raise PhysicsConfigurationError(
                f"Invalid {name}={value!r} for the detuning calculation. "
                "F05: no fallback is permitted."
            )

    # DEBUG: Print actual values being used
    print(f"DEBUG: Using probe_wavelength = {probe_wavelength} nm")
    print(f"DEBUG: Using coupling_wavelength = {coupling_wavelength} nm")
    print(f"DEBUG: Using rf_frequency = {rf_frequency/1e9:.3f} GHz")

    # Calculate each detuning
    delta_p = calculate_probe_detuning(
        atom, levels['ground'], levels['excited'], probe_wavelength
    )

    delta_c = calculate_coupling_detuning(
        atom, levels['excited'], levels['rydberg1'], coupling_wavelength
    )

    delta_rf = calculate_rf_detuning(
        atom, levels['rydberg1'], levels['rydberg2'], rf_frequency
    )

    logger.info(f"Calculated detunings: Δ_P={delta_p:.2f}rad/μs, "
               f"Δ_C={delta_c:.2f}rad/μs, Δ_RF={delta_rf:.2f}rad/μs")

    # F05: every produced detuning must be a finite number; a NaN/inf/None
    # result must never propagate into the Hamiltonian.
    for label, value in (('Delta_P', delta_p), ('Delta_C', delta_c),
                         ('Delta_RF', delta_rf)):
        if value is None:
            raise PhysicsCalculationError(
                f"{label} detuning is None (probe={probe_wavelength} nm, "
                f"coupling={coupling_wavelength} nm, f_RF={rf_frequency} Hz). "
                "F05: no fallback is permitted."
            )
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise PhysicsCalculationError(
                f"{label} detuning is not numeric ({value!r}). "
                "F05: no fallback is permitted."
            ) from exc
        if not np.isfinite(numeric):
            raise PhysicsCalculationError(
                f"{label} detuning is not finite ({numeric!r}) "
                f"(probe={probe_wavelength} nm, coupling={coupling_wavelength} nm, "
                f"f_RF={rf_frequency} Hz). F05: no fallback is permitted."
            )

    return delta_p, delta_c, delta_rf
