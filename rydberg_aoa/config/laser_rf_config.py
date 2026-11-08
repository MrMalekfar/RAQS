"""
Laser and RF parameter calculations.
"""

import math
import numpy as np
import scipy as sc
# --- Laser and RF Parameters (following Bussey et al. 2024 Figure 2) ---
# Import all parameters from parameters.py for centralized configuration
# This ensures all input parameters are defined in one place
from . import parameters as params
from ..arc.rabi_frequency import calculate_rabi_frequency as calc_rabi_laser, calculate_rf_rabi_frequency as calc_rabi_rf
from ..arc import LaserParams, RFParams, AtomicLevel
from ..exceptions import PhysicsConfigurationError

# ---------------------------------------------------------------------------
# BEAM-002 — SINGLE SOURCE OF TRUTH for the laser beam-size convention.
#
# Experimental source (Simons et al., APL 108, 174101 (2016), modeled by
# Bussey et al. 2024 §4.2 at the same operating point): the probe and
# coupling lasers are specified by their INTENSITY-FWHM FULL-WIDTH
# DIAMETERS —
#
#     probe D_FWHM        =  94 µm
#     coupling D_FWHM     = 144 µm
#
# TEM00 Gaussian convention:  I(r) = I0·exp(-2 r²/w0²)
#
#     D_FWHM = w0·√(2 ln 2)     =>     w0 = D_FWHM / √(2 ln 2)
#
# where w0 is the 1/e² INTENSITY RADIUS.  ARC ``getRabiFrequency`` (and
# ``getDrivingPower``) define ``laserWaist`` as exactly this radius.
#
# BEAM-001 finding: the historical Bussey 0.160/0.244 mm values are the
# 1/e² DIAMETERS (159.6725/244.6047 µm derived from the FWHM values above)
# and were previously consumed as ARC radii — a factor-of-two beam-radius
# error.  ``beam_waist_mm`` below is therefore the DERIVED 1/e² RADIUS,
# computed in ONE place by ``gaussian_fwhm_diameter_to_1e2_radius_mm`` from
# the FWHM source; the derived radii (≈79.836/122.302 µm) are never
# hard-coded elsewhere.
# ---------------------------------------------------------------------------
BEAM_FWHM_DIAMETER_UM = {
    'probe': 94.0,
    'coupling': 144.0,
}


def gaussian_fwhm_diameter_to_1e2_radius_mm(fwhm_diameter_um):
    """Convert an intensity-FWHM full-width DIAMETER to a 1/e² intensity RADIUS.

    w0 = D_FWHM / sqrt(2 ln 2)

    Args:
        fwhm_diameter_um: intensity-FWHM full width [µm] (experimental value)

    Returns:
        float: 1/e² intensity radius w0 [mm] (ARC ``laserWaist`` convention)
    """
    return (float(fwhm_diameter_um) / _FWHM_TO_1E2_RADIUS) / 1000.0


# w0 / D_FWHM = 1/√(2 ln 2) — Gaussian TEM00 intensity conversion factor
_FWHM_TO_1E2_RADIUS = math.sqrt(2.0 * math.log(2.0))

# Laser parameters (read from parameters.py)
LASER_PARAMS = {
    'probe': {
        # OP-001 — FROZEN SENSOR OPERATING POINT (HET-001-selected):
        #   CURRENT SENSOR OPERATING POINT:  P_probe = 0.10 µW = 0.0001 mW
        #   REFERENCE / SIMONS INPUT:        P_probe = 17.5 µW — retained ONLY
        #     as a documented reference profile (at that power the AT doublet
        #     is power-broadened away and heterodyne depth falls ~80x; it must
        #     never be the production sensor default).
        # Selected within the CURRENT effective four-level model; not claimed
        # as a universal physical optimum (see POL/DEN caveats).
        'power_mw': 0.0001,
        # BEAM-002: derived 1/e² intensity RADIUS [mm] from the Simons probe
        # FWHM full-width 94 µm (helper above). Consumed as ARC ``laserWaist``
        # (radius [m] after ×1e-3) by both the Rabi path and the transit-rate
        # model. This is a RADIUS — never a diameter.
        'beam_waist_mm': gaussian_fwhm_diameter_to_1e2_radius_mm(
            BEAM_FWHM_DIAMETER_UM['probe']),
        'wavelength_nm': 780.241476,   # Rb D2 line
        'detuning_mhz': 0.0,      # On resonance
        'polarization': 0         # Linear polarization (works for ground-excited)
    },
    'coupling': {
        'power_mw': 32,         # 32 mW — coupling laser (project operating point)
        # BEAM-002: derived 1/e² intensity RADIUS [mm] from the Simons coupling
        # FWHM full-width 144 µm (helper above). RADIUS — never a diameter.
        'beam_waist_mm': gaussian_fwhm_diameter_to_1e2_radius_mm(
            BEAM_FWHM_DIAMETER_UM['coupling']),
        'wavelength_nm': 483.4,   # Rydberg coupling wavelength
        'detuning_mhz': 0.0,      # On resonance
        'polarization': 0         # Linear polarization
    }
}
# RF field parameters for heterodyne sensing (read from parameters.py)
# Signal RF: Weaker field to be sensed (modulated for AoA detection)
SIGNAL_RF_PARAMS = {
    'frequency_hz': params.RF_SIGNAL_FREQUENCY,
    'electric_field_vm': params.RF_SIGNAL_E_FIELD,
    'polarization': 0            # Linear polarization
}

# Local Oscillator RF: Stronger field for heterodyne mixing
LO_RF_PARAMS = {
    'frequency_hz': params.RF_LO_FREQUENCY,
    'electric_field_vm': params.RF_LO_E_FIELD,
    'polarization': 0            # Linear polarization
}

# Backward compatibility: RF_PARAMS is a separate LO configuration copy.
# LAB-002: configure_rf_fields() publishes both views in place;
# read_configured_rf_fields() rejects disagreement instead of guessing.
# (Used for level optimization - typically the LO frequency determines the Rydberg transition)
RF_PARAMS = {**LO_RF_PARAMS}


def calculate_laser_rabi_frequency(atom, level1, level2, laser_params):
    """
    Calculate Rabi frequency for laser transition using ARC's direct method.
    Delegates to the optimized rabi_frequency module.

    Parameters:
    - atom: ARC atom object
    - level1, level2: dictionaries with 'n', 'l', 'j' keys
    - laser_params: dictionary with power, beam_waist, wavelength, etc.

    Returns:
    - Omega in rad/s
    """
    if atom is None:
        raise PhysicsConfigurationError(
            "ARC atom is unavailable; the probe/coupling laser Rabi "
            f"frequency for {getattr(level2, 'label', level2)} cannot be "
            "calculated. F05: no fallback is permitted."
        )

    # Convert dictionary parameters to AtomicLevel and LaserParams objects
    from_level = AtomicLevel(**level1) if isinstance(level1, dict) else level1
    to_level = AtomicLevel(**level2) if isinstance(level2, dict) else level2
    laser_params_obj = LaserParams(**laser_params) if isinstance(laser_params, dict) else laser_params

    # C-001 unit contract: rabi_frequency helpers return the ANGULAR Rabi
    # frequency in rad/s. Exactly ONE s->us conversion (x1e-6) is applied
    # in atomic_config. There is NO 2pi conversion anywhere in this chain:
    # Omega [rad/us] = 2pi * f_Rabi [MHz] is a display relation only.
    return calc_rabi_laser(atom, to_level, laser_params_obj, from_level=from_level)


def calculate_rf_rabi_frequency(atom, level1, level2, rf_params):
    """
    Calculate Rabi frequency for RF transition using ARC's direct method.
    Delegates to the optimized rabi_frequency module.

    Parameters:
    - atom: ARC atom object
    - level1, level2: dictionaries with 'n', 'l', 'j' keys
    - rf_params: dictionary with frequency, electric_field, etc.

    Returns:
    - Omega in rad/s
    """
    if atom is None:
        raise PhysicsConfigurationError(
            "ARC atom is unavailable; the RF Rabi frequency for "
            f"{getattr(level2, 'label', level2)} cannot be calculated. "
            "F05: no fallback is permitted."
        )

    # Convert dictionary parameters to AtomicLevel and RFParams objects
    level1_obj = AtomicLevel(**level1) if isinstance(level1, dict) else level1
    level2_obj = AtomicLevel(**level2) if isinstance(level2, dict) else level2
    rf_params_obj = RFParams(**rf_params) if isinstance(rf_params, dict) else rf_params

    # C-001 unit contract: see calculate_laser_rabi_frequency above. The
    # helper returns the angular Rabi frequency in rad/s; exactly ONE
    # s->us conversion happens in atomic_config; no 2pi anywhere.
    return calc_rabi_rf(atom, level1_obj, level2_obj, rf_params_obj)


def get_signal_rf_rabi_frequency(atom, level1, level2, rf_params):
    """
    Calculate signal RF Rabi frequency using SIGNAL_RF_PARAMS.

    Returns:
    - Omega in rad/s
    """
    if atom is None:
        raise PhysicsConfigurationError(
            "ARC atom is unavailable; the signal-RF Rabi frequency for "
            f"{getattr(level2, 'label', level2)} cannot be calculated. "
            "F05: no fallback is permitted."
        )
    # Convert dictionary parameters to AtomicLevel and RFParams objects
    level1_obj = AtomicLevel(**level1) if isinstance(level1, dict) else level1
    level2_obj = AtomicLevel(**level2) if isinstance(level2, dict) else level2
    rf_params_obj = RFParams(**rf_params) if isinstance(rf_params, dict) else rf_params
    # C-001 unit contract: see calculate_laser_rabi_frequency above. The
    # helper returns the angular Rabi frequency in rad/s; exactly ONE
    # s->us conversion happens in atomic_config; no 2pi anywhere.
    return calc_rabi_rf(atom, level1_obj, level2_obj, rf_params_obj)
