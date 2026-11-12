"""
Physical constants and simulation parameters.
"""

import numpy as np
import scipy as sc
from .atom_species import get_arc_atom, LEGACY_ATOM

# F05 — fail-fast physics exception hierarchy: no silent density/physics
# fallbacks are permitted anywhere in this module.
from ..exceptions import PhysicsError, PhysicsCalculationError

# --- Unit Conversion Factor ---
# Factor: (rad/us) * (1e6 us/s) = (rad/s)
US_TO_S_FACTOR = 1e6

# --- Atomic Level Selection Tolerances ---
# Tolerances for finding atomic transitions (can be tuned for accuracy vs speed)
LEVEL_SELECTION_TOLERANCES = {
    'wavelength_tolerance_nm': 5.0,      # ±5 nm for laser wavelengths
    'frequency_tolerance_hz': 0.3e9,       # ±5 GHz for RF frequencies (tightened for speed)
    'min_dipole_threshold': 1e-32,       # Minimum dipole for significant transitions
    'n_range_min': 10,                   # Minimum principal quantum number to search
    'n_range_max': 110,                   # Maximum principal quantum number to search
}

# --- Calculated System Detunings (in rad/us) ---
# Detunings are dynamically calculated based on level optimization results
# These replace the hardcoded values following the "no hardcoded parameters" principle

# Detunings are now accessed from atomic_config where they belong

# Decoherence rates
GAMMA_DEPHASING = 2 * np.pi * 0.5 # Transit/laser/collision dephasing
GAMMA_2 = 2 * np.pi * 0.1   # Effective single-value Decay Rate gamma_2 (e.g., 0.1 MHz)

# Individual dephasing components for analytical solutions
GAMMA_COLLISION_DEPHASING = 2 * np.pi * 0.2  # Collision dephasing
GAMMA_TRANSIT_DEPHASING = 2 * np.pi * 0.2    # Transit dephasing

# Laser linewidth broadening (from Bussey et al. 2024)
GAMMA_PROBE_LINEWIDTH = 2 * np.pi * 0.001    # Probe laser linewidth (1 kHz)
GAMMA_COUPLING_LINEWIDTH = 2 * np.pi * 0.001 # Coupling laser linewidth (1 kHz)
GAMMA_RF_LINEWIDTH = 2 * np.pi * 0.0         # RF field linewidth (0, as per Bussey et al.)

# --- Vapor Cell Temperature ---
VAPOR_CELL_TEMPERATURE_K = 300.15  # 27°C (typical heated vapor cell)

# --- Doppler Broadening (thermal velocity-class averaging) ---
# Ensemble-averaged EIT response: ⟨T⟩ = Σ_i w_i · T(Δ + Δ_D(v_i)) over
# Maxwell-Boltzmann velocity classes (see rydberg_aoa/physics/doppler/)
DOPPLER_ENABLED = True              # Enable velocity-class averaging in steady-state spectrum
DOPPLER_N_CLASSES = 9               # Legacy fixed-class quadrature nodes (GH/uniform only)
DOPPLER_METHOD = 'adaptive'         # 'adaptive' (production) | 'gauss_hermite' | 'uniform'
DOPPLER_N_SIGMA = 5.0               # Maxwell truncation half-width in units of σ_v (adaptive)
DOPPLER_ADAPTIVE_RTOL = 1e-6        # Adaptive quadrature relative tolerance (complex rho)
DOPPLER_ADAPTIVE_ATOL = 1e-12       # Adaptive quadrature absolute tolerance
DOPPLER_COUNTER_PROPAGATING = True  # Probe/coupling counter-propagating geometry (intended EIT vapour-cell setup)

# --- PHY-SW-001: physical-effect switches -----------------------------------
# Independent enable/disable of the environmental broadening mechanisms in the
# NUMERICAL model, so numerical and analytical solutions can later solve the
# SAME physical problem (analytical parity profile, see
# rydberg_aoa/simulation/effect_switches.py).
#
# All True = the currently accepted production model (default invariance):
# every default-path result is bit-identical to pre-PHY-SW-001 behavior.
# There is deliberately NO RF switch: RF-off remains the Omega_RF = 0 contract.
# Spontaneous decay, power broadening, EIT/AT, cell propagation, density and
# the coupling laser are structural model elements and are NOT switchable.
TRANSIT_ENABLED = True               # DPH-002B reset collapse operators (loss/repopulation)
LASER_LINEWIDTH_DEPHASING_ENABLED = True   # gamma_laser = {probe,coupling,rf} linewidth term
COLLISION_DEPHASING_ENABLED = True         # gamma_collision = n*sigma*<v_rel> term
RB85_MASS_KG = 1.41e-25             # Mass of one Rb-85 atom [kg]

# --- Vapor Cell Parameters ---
# REF-002: 75 mm REFERENCE vapor-cell interaction length (Simons et al. 2016
# geometry; supersedes the arbitrary 100 mm owner value).
CELL_LENGTH = 0.075  # Vapor cell length in meters (75 mm)

# --- Steady-State Spectrum Scan ---
SCAN_RANGE_MHZ = 20.0
N_SCAN = 201

# --- Heterodyne RF Configuration ---
# Signal RF (electromagnetic field to be detected)
RF_SIGNAL_FREQUENCY = 132.6491e9
RF_SIGNAL_E_FIELD = 0.005              # V/m (electric field strength)

# Local Oscillator RF (for heterodyne mixing)
RF_LO_FREQUENCY = 132.649e9              # (LO frequency)
# OP-001 — FROZEN SENSOR OPERATING POINT (HET-001-selected):
#   E_LO = 1.5 V/m  (production SENSOR LO field for the CURRENT effective
#   four-level model — NOT a universal physical optimum and NOT the Simons
#   reference). The weak signal stays RF_SIGNAL_E_FIELD = 0.005 V/m; the two
#   fields are distinct by construction (Omega_LO / Omega_signal = 300).
RF_LO_E_FIELD = 1.5

# --- Array & RF Scenario ---
N_ELEMENTS = 6
SPACING = sc.constants.c / RF_SIGNAL_FREQUENCY / 8  # lamda/4 spacing                  # V/m (stronger LO field)
# Target Angle of Arrival (AoA)
RF_FREQUENCY = RF_SIGNAL_FREQUENCY     # RF frequency for RF signal
TARGET_AOA_DEG = -30.0   # True AoA for simulation
# Computed IF frequency: |f_signal - f_LO| (heterodyne physics)
IF_FREQ = (RF_SIGNAL_FREQUENCY - RF_LO_FREQUENCY) / 1e6  # MHz, SIGNED (f_signal - f_LO); +0.1 for the current pair (RF-DIR-001D)
# print(f"Computed IF frequency: {IF_FREQ:.3f} MHz (signed f_signal - f_LO)")

# --- OP-001 — SENSOR OPERATING POINT: probe operating detuning ---
# Concept separation (A vs B):
#   A. Spectrum-scan detunings: stay at the level-optimization center
#      (used by simulate_spectrum / run_steady_state_spectrum for EIT/AT
#      plots; SCAN_RANGE_MHZ / N_SCAN define the grid).
#   B. Sensor operating detuning: applied ONLY to the time-domain /
#      heterodyne sensing path (SimulationRunner._sensor_probe_detunings).
# Unit contract (C-001/OP-001): Delta_sensor [rad/us] = 2*pi * [MHz].
# Exactly ONE 2*pi factor — NO extra 2*pi.
SENSOR_PROBE_DETUNING_MHZ = -0.5   # frozen sensor probe operating detuning

# --- ARC-Calculated Parameters ---
# Calculate atomic density using ARC temperature-dependent vapor pressure
def calculate_atomic_density_with_arc():
    """
    Calculate rubidium vapor density at given temperature using ARC.

    Returns:
        float: Atomic number density in m^-3

    Raises:
        PhysicsCalculationError: If ARC cannot provide the vapor density.
            F05 (F05-16): the historical hard-coded default-density constant
            is removed; a failed density calculation stops the run instead of
            feeding a fabricated number density into the susceptibility model.
    """
    try:
        # This explicit calculation uses the selected atom when configured.
        import sys
        from .atom_species import arc_density
        ac = sys.modules.get(__package__ + '.atomic_config')
        if ac is not None and ac.ATOM is not None:
            atom, spec = ac.ATOM, ac.ATOM_SPEC
        else:
            atom, spec = get_arc_atom(LEGACY_ATOM), LEGACY_ATOM
        density = arc_density(atom, spec, VAPOR_CELL_TEMPERATURE_K)
        #print(f"ARC: Calculated atomic density at {VAPOR_CELL_TEMPERATURE_K:.1f}K: {density:.2e} m^-3")
        return density
    except PhysicsError:
        raise
    except Exception as e:
        # F05: no default density. N0 enters the Beer-Lambert exponent, so a
        # fabricated number density would silently rescale every transmission.
        raise PhysicsCalculationError(
            f"ARC vapor density calculation failed at "
            f"{VAPOR_CELL_TEMPERATURE_K} K: {type(e).__name__}: {e}. "
            "F05: no hard-coded density fallback."
        ) from e

# Published by explicit atomic configuration.
CALCULATED_N0 = None

# --- Heterodyne Time-Domain Simulation ---
# Rabi frequencies will be calculated by atomic_config.py using ARC dipole moments
# Default fallback values (will be overridden by physics-based calculations)
OMEGA_S_BASE = 2 * np.pi * 0.314  # ~0.314 MHz fallback (will be updated by atomic_config.py)
OMEGA_LO_BASE = 2 * np.pi * 0.948 # ~0.948 MHz fallback (will be updated by atomic_config.py)

# --- SI Constants & Susceptibility Parameters (SI Units) ---
# NOTE: These are for the analytical transmission model (Eq. 5, 6)
class SI_Constants:
    """Class to hold constants for susceptibility calculation (SI units)."""
    # Speed of light (m/s)
    LIGHT_SPEED = sc.constants.c  # 299792458.0
    # Vacuum permittivity (F/m)
    EPSILON_0 = sc.constants.epsilon_0  # 8.854e-12
    # Reduced Planck constant (J*s)
    HBAR = sc.constants.h / (2*np.pi)  # 1.0545718e-34

    # Atomic density (atoms/m^3) - calculated from ARC temperature-dependent vapor pressure
    N0 = CALCULATED_N0

    # Probe laser wavelength (m) - Rubidium D2 line is 780 nm
    LAMBDA_P = 7.80241476e-7   # 780.241476 nm — Rb D2 line

# Time vector (in microseconds)
T_MAX = 15 / abs(IF_FREQ)  # 15 cycles of the IF beat note (record length: magnitude use, RF-DIR-001D)
T_STEPS = 1001
ATOMIC_TRANSIENT_DELAY_US = 40.0
