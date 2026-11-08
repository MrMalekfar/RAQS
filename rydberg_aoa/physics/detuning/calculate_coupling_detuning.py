"""
Calculate coupling laser detuning.
"""

import logging
from ._calculate_transition_frequency import _calculate_transition_frequency
from ._wavelength_to_frequency import _wavelength_to_frequency
from ._hz_to_rad_per_us import _hz_to_rad_per_us

logger = logging.getLogger(__name__)


def calculate_coupling_detuning(atom, excited_level, rydberg1_level, coupling_wavelength_nm: float) -> float:
    """
    Calculate coupling laser detuning: Δ_C = ω_coupling_laser - ω_(excited → rydberg1)

    Args:
        atom: ARC atom object
        excited_level, rydberg1_level: AtomicLevel objects for 5P3/2 → nd transition
        coupling_wavelength_nm: Actual coupling laser wavelength (nm)

    Returns:
        float: Detuning in rad/μs for Hamiltonian (Δ_C in rotating frame)
    """
    # Calculate resonant transition frequency from ARC energies
    omega_resonant_hz = _calculate_transition_frequency(atom, excited_level, rydberg1_level)

    # Calculate laser frequency from wavelength
    omega_laser_hz = _wavelength_to_frequency(coupling_wavelength_nm)

    # Detuning: Δ_C = ω_laser - ω_resonant
    delta_freq_hz = omega_laser_hz - omega_resonant_hz

    # Convert to rad/μs for Hamiltonian
    delta_rad_us = _hz_to_rad_per_us(delta_freq_hz)

    logger.debug(f"Coupling detuning: ω_laser={omega_laser_hz/1e9:.3f}GHz, "
                f"ω_resonant={omega_resonant_hz/1e9:.3f}GHz, "
                f"Δ={delta_rad_us:.2f}rad/μs ({delta_freq_hz/1e3:.1f}kHz)")

    return delta_rad_us
