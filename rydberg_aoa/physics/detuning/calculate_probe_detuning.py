"""
Calculate probe laser detuning.
"""

import logging
from . import _calculate_transition_frequency
from ._calculate_transition_frequency import _calculate_transition_frequency
from ._wavelength_to_frequency import _wavelength_to_frequency
from ._hz_to_rad_per_us import _hz_to_rad_per_us

logger = logging.getLogger(__name__)


def calculate_probe_detuning(atom, ground_level, excited_level, probe_wavelength_nm: float) -> float:
    """
    Calculate probe laser detuning: Δ_P = ω_probe_laser - ω_(ground → excited)

    Args:
        atom: ARC atom object
        ground_level, excited_level: AtomicLevel objects for 5S1/2 → 5P3/2 transition
        probe_wavelength_nm: Actual probe laser wavelength (nm)

    Returns:
        float: Detuning in rad/μs for Hamiltonian (Δ_P in rotating frame)
    """
    # Calculate resonant transition frequency from ARC energies
    omega_resonant_hz = _calculate_transition_frequency(atom, ground_level, excited_level)

    # Calculate laser frequency from wavelength
    omega_laser_hz = _wavelength_to_frequency(probe_wavelength_nm)

    # Detuning: Δ_P = ω_laser - ω_resonant
    delta_freq_hz = omega_laser_hz - omega_resonant_hz

    # Convert to rad/μs for Hamiltonian
    delta_rad_us = _hz_to_rad_per_us(delta_freq_hz)

    logger.debug(f"Probe detuning: ω_laser={omega_laser_hz/1e9:.3f}GHz, "
                f"ω_resonant={omega_resonant_hz/1e9:.3f}GHz, "
                f"Δ={delta_rad_us:.2f}rad/μs ({delta_freq_hz/1e3:.1f}kHz)")

    return delta_rad_us
